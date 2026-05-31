import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from model import RecurrentAnatomyGATNew_A


# ==============================================================================
# 1. 核心配置
# ==============================================================================

DEFAULT_MODEL_PATH = 'checkpoints/AnatomyGATCorrectionCE_Visual_Edge_Geom_Prior_T5/best_model.pth'
DEFAULT_TEST_DATA_PATH = 'gnn_data/test_ccsw_sin_arch4_edge_01.pt'
DEFAULT_OUTPUT_NAME = 'graph_by_graph_predictions_recurrent_ccsw_i5_visual_correction.pt'

N_CLASSES = 49
BACKGROUND_IDX = 48
NUM_RELATIONS = 4
NUM_ITERATIONS = 5
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RELABEL_MARGIN = 0.02
RELABEL_MIN_PROB = 0.12
DUPLICATE_IOU_THRESHOLD = 0.50
DUPLICATE_IOS_THRESHOLD = 0.75

CONFUSION_SET_CLASSES = {
    11, 12,
    21, 22,
    31, 32,
    41, 42,
    51, 52,
    61, 62,
    71, 72,
    81, 82,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the recurrent anatomy GNN as a conservative tooth-number corrector.'
    )
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--test-data-path', default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument('--output-name', default=DEFAULT_OUTPUT_NAME)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--relabel-margin', type=float, default=RELABEL_MARGIN)
    parser.add_argument('--relabel-min-prob', type=float, default=RELABEL_MIN_PROB)
    parser.add_argument('--duplicate-iou-threshold', type=float, default=DUPLICATE_IOU_THRESHOLD)
    parser.add_argument('--duplicate-ios-threshold', type=float, default=DUPLICATE_IOS_THRESHOLD)
    parser.set_defaults(use_visual=True)
    parser.add_argument('--use-visual', dest='use_visual', action='store_true', help='Enable visual features.')
    parser.add_argument('--no-visual', dest='use_visual', action='store_false', help='Disable visual features.')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, confusion_set):
    """计算总体准确率和混淆集准确率。"""
    overall_acc = accuracy_score(y_true, y_pred)

    mask = np.isin(y_true, list(confusion_set))
    total_confusion_nodes = mask.sum()

    if total_confusion_nodes == 0:
        confusion_set_acc = 0.0
    else:
        correct_confusion_nodes = (y_true[mask] == y_pred[mask]).sum()
        confusion_set_acc = correct_confusion_nodes / total_confusion_nodes

    return overall_acc, confusion_set_acc


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """绘制并保存混淆矩阵。"""
    labels = range(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_prior_labels(graph, n_classes=N_CLASSES):
    if hasattr(graph, 'pred_labels_raw'):
        return graph.pred_labels_raw.long()
    return graph.x_prior[:, :n_classes].argmax(dim=1).long()


def get_detector_scores(graph):
    if hasattr(graph, 'pred_scores_raw'):
        return graph.pred_scores_raw.float()
    return ((graph.x_prior[:, -1].float() + 1.0) / 2.0).clamp(0.0, 1.0)


def get_raw_boxes(graph):
    if hasattr(graph, 'pred_bboxes_raw'):
        return graph.pred_bboxes_raw.float()
    return graph.pos.float()


def get_scalar(value, default=-1):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].detach().cpu().item())
    return int(value)


def box_iou_one_to_many(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp_min(0)
    inter_h = (y2 - y1).clamp_min(0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    union = area_box + area_boxes - inter
    return inter / union.clamp_min(1e-6)


def box_ios_one_to_many(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp_min(0)
    inter_h = (y2 - y1).clamp_min(0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    smaller_area = torch.minimum(area_box.expand_as(area_boxes), area_boxes)
    return inter / smaller_area.clamp_min(1e-6)


def conservative_relabel(logits, raw_labels, relabel_margin, relabel_min_prob):
    """
    默认保留 detector 原标签；只有最佳非背景类明显强于原标签时才改编号。
    background 不直接删框。
    """
    probs = F.softmax(logits, dim=1)
    raw_labels = raw_labels.to(logits.device).clamp(0, N_CLASSES - 1)

    best_tooth_probs, best_tooth_labels = probs[:, :BACKGROUND_IDX].max(dim=1)
    raw_probs = probs.gather(1, raw_labels.unsqueeze(1)).squeeze(1)

    relabel_mask = (
        (best_tooth_labels != raw_labels)
        & (best_tooth_probs >= relabel_min_prob)
        & ((best_tooth_probs - raw_probs) >= relabel_margin)
    )

    final_labels = raw_labels.clone()
    final_labels[relabel_mask] = best_tooth_labels[relabel_mask]
    return final_labels, probs, best_tooth_labels, relabel_mask


def duplicate_cleanup(labels, boxes, detector_scores, probs, duplicate_iou_threshold, duplicate_ios_threshold):
    """
    只清理同一最终牙位的强重叠重复框；这一步是唯一会把框置为 background 的后处理。
    """
    if labels.numel() <= 1:
        return labels, torch.zeros_like(labels, dtype=torch.bool)

    labels = labels.clone()
    boxes = boxes.to(labels.device)
    detector_scores = detector_scores.to(labels.device)
    duplicate_mask = torch.zeros_like(labels, dtype=torch.bool)

    label_probs = probs.gather(1, labels.clamp(0, N_CLASSES - 1).unsqueeze(1)).squeeze(1)
    keep_score = detector_scores + 0.25 * label_probs

    for cls_idx in range(BACKGROUND_IDX):
        cls_indices = torch.where(labels == cls_idx)[0]
        if cls_indices.numel() <= 1:
            continue

        order = cls_indices[torch.argsort(keep_score[cls_indices], descending=True)]
        kept = []
        for idx in order:
            if not kept:
                kept.append(idx)
                continue

            kept_tensor = torch.stack(kept)
            ious = box_iou_one_to_many(boxes[idx], boxes[kept_tensor])
            ios = box_ios_one_to_many(boxes[idx], boxes[kept_tensor])
            if bool(((ious >= duplicate_iou_threshold) | (ios >= duplicate_ios_threshold)).any()):
                labels[idx] = BACKGROUND_IDX
                duplicate_mask[idx] = True
            else:
                kept.append(idx)

    return labels, duplicate_mask


def calibrate_scores(detector_scores, final_labels, probs):
    """
    Keep detector scores as the main confidence and apply a light GNN confidence calibration.
    """
    safe_labels = final_labels.clamp(0, N_CLASSES - 1)
    label_probs = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    bg_probs = probs[:, BACKGROUND_IDX]
    confidence_delta = (label_probs - bg_probs).clamp(-1.0, 1.0)
    multiplier = (1.0 + 0.15 * confidence_delta).clamp(0.85, 1.15)
    return (detector_scores * multiplier).clamp(0.0, 1.0)


def postprocess_graph(logits, graph, relabel_margin, relabel_min_prob, duplicate_iou_threshold, duplicate_ios_threshold):
    raw_labels = get_prior_labels(graph).to(logits.device)
    detector_scores = get_detector_scores(graph).to(logits.device)
    boxes = get_raw_boxes(graph).to(logits.device)

    relabeled, probs, best_tooth_labels, relabel_mask = conservative_relabel(
        logits, raw_labels, relabel_margin, relabel_min_prob
    )
    postprocessed, duplicate_mask = duplicate_cleanup(
        relabeled, boxes, detector_scores, probs, duplicate_iou_threshold, duplicate_ios_threshold
    )
    calibrated_scores = calibrate_scores(detector_scores, postprocessed, probs)
    return {
        'pred_original': raw_labels.detach().cpu(),
        'pred_model_best_tooth': best_tooth_labels.detach().cpu(),
        'pred_postprocessed': postprocessed.detach().cpu(),
        'pred_scores_gnn': calibrated_scores.detach().cpu(),
        'gnn_probs': probs.detach().cpu(),
        'gnn_logits': logits.detach().cpu(),
        'relabel_mask': relabel_mask.detach().cpu(),
        'duplicate_suppressed_mask': duplicate_mask.detach().cpu(),
    }


def unpack_logits(model_output):
    all_step_logits = model_output[0] if isinstance(model_output, tuple) else model_output
    return all_step_logits[-1] if isinstance(all_step_logits, list) else all_step_logits


@torch.no_grad()
def generate_per_graph_predictions(model, test_data_list, device, save_path, args):
    """生成逐图(per-graph)预测文件。"""
    logging.info("--- 正在生成逐图预测文件... ---")

    loader = DataLoader(test_data_list, batch_size=1, shuffle=False)
    model.eval()
    all_graph_results = []

    for batch in loader:
        batch = batch.to(device)
        graph = batch.to_data_list()[0]

        img_id = get_scalar(getattr(graph, 'img_id', None))
        img_path = graph.img_path[0] if isinstance(graph.img_path, list) else graph.img_path

        out_gnn = model(batch, return_att=True)
        final_logits = unpack_logits(out_gnn)
        att_weights, edge_index, edge_type = None, None, None
        if isinstance(out_gnn, tuple) and len(out_gnn) >= 4:
            _, att_weights, edge_index, edge_type = out_gnn

        processed = postprocess_graph(
            final_logits,
            graph,
            args.relabel_margin,
            args.relabel_min_prob,
            args.duplicate_iou_threshold,
            args.duplicate_ios_threshold
        )

        y_true = graph.y.long().cpu()
        raw_boxes = get_raw_boxes(graph).cpu().numpy()
        raw_scores = get_detector_scores(graph).cpu()

        graph_result_dict = {
            'img_id': img_id,
            'img_path': img_path,
            'y_true': y_true,
            'pred_original': processed['pred_original'],
            'pred_gnn': processed['pred_postprocessed'],
            'pred_postprocessed': processed['pred_postprocessed'],
            'pred_model_best_tooth': processed['pred_model_best_tooth'],
            'gnn_logits': processed['gnn_logits'],
            'gnn_probs': processed['gnn_probs'],
            'relabel_mask': processed['relabel_mask'],
            'duplicate_suppressed_mask': processed['duplicate_suppressed_mask'],
            'pred_scores_raw': raw_scores,
            'pred_scores_gnn': processed['pred_scores_gnn'],
            'edge_index': edge_index.cpu().numpy() if edge_index is not None else None,
            'att_weights': att_weights.cpu().numpy() if att_weights is not None else None,
            'edge_types': edge_type.cpu().numpy() if edge_type is not None else None,
            'raw_boxes': raw_boxes,
        }
        all_graph_results.append(graph_result_dict)

    try:
        torch.save(all_graph_results, save_path)
        logging.info(f"逐图预测文件已保存到: {save_path}")
    except Exception as e:
        logging.error(f"保存出错: {e}")


@torch.no_grad()
def evaluate_aggregate(model, loader, device, args):
    """计算整个测试集的聚合指标，并测量网络推理速度。"""
    model.eval()
    all_y_true = []
    all_pred_original = []
    all_pred_gnn = []
    relabel_total = 0
    duplicate_suppressed_total = 0

    total_inference_time = 0.0
    total_graphs = 0

    if device.type == 'cuda':
        dummy_batch = next(iter(loader)).to(device)
        _ = model(dummy_batch)
        torch.cuda.synchronize()

    logging.info("开始批量评估与测速...")

    for batch in loader:
        batch = batch.to(device)
        num_graphs_in_batch = batch.num_graphs

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        out_gnn = model(batch)
        final_logits = unpack_logits(out_gnn)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        total_inference_time += end_time - start_time
        total_graphs += num_graphs_in_batch

        node_offset = 0
        for graph in batch.to_data_list():
            num_nodes = graph.num_nodes
            single_logits = final_logits[node_offset: node_offset + num_nodes]
            processed = postprocess_graph(
                single_logits,
                graph,
                args.relabel_margin,
                args.relabel_min_prob,
                args.duplicate_iou_threshold,
                args.duplicate_ios_threshold
            )

            all_y_true.append(graph.y.long().cpu().numpy())
            all_pred_original.append(processed['pred_original'].numpy())
            all_pred_gnn.append(processed['pred_postprocessed'].numpy())
            relabel_total += int(processed['relabel_mask'].sum())
            duplicate_suppressed_total += int(processed['duplicate_suppressed_mask'].sum())
            node_offset += num_nodes

    y_true = np.concatenate(all_y_true)
    pred_original = np.concatenate(all_pred_original)
    pred_gnn = np.concatenate(all_pred_gnn)

    avg_latency_ms = (total_inference_time / total_graphs) * 1000 if total_graphs > 0 else 0.0
    fps = total_graphs / total_inference_time if total_inference_time > 0 else 0.0

    logging.info("\n推理速度统计")
    logging.info(f"总耗时: {total_inference_time:.4f}s | 处理图数: {total_graphs}")
    logging.info(f"平均延迟 (Latency): {avg_latency_ms:.4f} ms/image")
    logging.info(f"吞吐量 (Throughput): {fps:.2f} FPS")
    logging.info(f"保守 relabel 节点数: {relabel_total}")
    logging.info(f"duplicate cleanup 抑制节点数: {duplicate_suppressed_total}")

    return y_true, pred_original, pred_gnn, fps, avg_latency_ms


def build_model(args):
    return RecurrentAnatomyGATNew_A(
        n_classes=N_CLASSES,
        num_relations=NUM_RELATIONS,
        num_iterations=args.num_iterations,
        use_visual=args.use_visual,
        use_geom=True,
        use_prior=True,
        use_edge_features=True,
        spatial_only=False
    ).to(DEVICE)


def main():
    args = parse_args()
    results_save_dir = os.path.dirname(args.model_path) or '.'
    os.makedirs(results_save_dir, exist_ok=True)

    log_file_path = os.path.join(results_save_dir, 'evaluation_recurrent.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"使用设备: {DEVICE}")
    logging.info(
        f"评估模型: RecurrentAnatomyGATNew_A "
        f"(Relations={NUM_RELATIONS}, Iterations={args.num_iterations}, use_visual={args.use_visual})"
    )
    logging.info(f"测试数据文件: {args.test_data_path}")
    logging.info(
        f"后处理参数: relabel_margin={args.relabel_margin}, "
        f"relabel_min_prob={args.relabel_min_prob}, duplicate_iou={args.duplicate_iou_threshold}, "
        f"duplicate_ios={args.duplicate_ios_threshold}"
    )

    if not os.path.exists(args.model_path):
        logging.error(f"模型文件未找到: {args.model_path}")
        return

    model = build_model(args)
    try:
        state = torch.load(args.model_path, map_location=DEVICE)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        logging.info(f"模型权重已加载: {args.model_path}")
    except Exception as e:
        logging.error(f"加载权重失败: {e}")
        return

    try:
        test_data_list = torch.load(args.test_data_path, map_location='cpu', weights_only=False)
        logging.info(f"测试数据已加载: {len(test_data_list)} 张图")
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return

    test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)

    y_true, pred_original, pred_gnn, fps, latency = evaluate_aggregate(
        model, test_loader, DEVICE, args
    )

    logging.info("\n--- 计算核心指标 ---")
    metrics_orig = calculate_metrics(y_true, pred_original, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true, pred_gnn, CONFUSION_SET_CLASSES)

    logging.info("\n" + "=" * 70)
    logging.info(f"{'指标':<24} | {'Detector':<15} | {'GNN corrected':<15}")
    logging.info("-" * 70)
    logging.info(f"{'总体准确率':<24} | {metrics_orig[0]:.4f}          | {metrics_gnn[0]:.4f}")
    logging.info(f"{'混淆集准确率':<24} | {metrics_orig[1]:.4f}          | {metrics_gnn[1]:.4f}")
    logging.info("-" * 70)
    logging.info(f"{'推理速度 (FPS)':<24} | {'-':<15} | {fps:.2f}")
    logging.info(f"{'单张延迟 (ms)':<24} | {'-':<15} | {latency:.2f}")
    logging.info("=" * 70 + "\n")

    simple_class_names = [str(i) for i in range(N_CLASSES)]
    report_gnn = classification_report(y_true, pred_gnn, target_names=simple_class_names, zero_division=0)
    logging.info("RecurrentGAT 分类报告预览:\n" + report_gnn)

    with open(os.path.join(results_save_dir, 'classification_report_recurrent.txt'), 'w') as f:
        f.write(report_gnn)

    plot_confusion_matrix(
        y_true, pred_gnn, simple_class_names,
        f'Confusion Matrix - RecurrentGAT Corrector (T={args.num_iterations})',
        os.path.join(results_save_dir, 'cm_recurrent_gat.png')
    )

    save_pt_path = os.path.join(results_save_dir, args.output_name)
    generate_per_graph_predictions(model, test_data_list, DEVICE, save_pt_path, args)

    logging.info("迭代模型评估结束。")


if __name__ == '__main__':
    main()
