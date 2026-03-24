import logging
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torch_geometric.data import DataLoader
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import RecurrentAnatomyGATNew, RecurrentAnatomyGATNew_A
from util import enforce_one_to_one_matching

# ==============================================================================
# 1. 核心配置
# ==============================================================================

MODEL_PATH = 'checkpoints/2026-02-22_10-15-26/best_model.pth'

TEST_DATA_PATH = 'gnn_data/ccsw_60_final_gnn.pt'  # 请确保这是包含 V4.1+ 结构的新数据
N_CLASSES = 49
NUM_RELATIONS = 4
NUM_ITERATIONS = 5  # [!! 新增 !!] 必须与训练时设置的迭代次数一致
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 结果保存目录
RESULTS_SAVE_DIR = os.path.dirname(MODEL_PATH)

# 混淆集定义 (监控核心指标)
CONFUSION_SET_CLASSES = {
    11, 12,  # 上颌右侧切牙
    21, 22,  # 上颌左侧切牙
    31, 32,  # 下颌左侧切牙
    41, 42,  # 下颌右侧切牙
    51, 52,
    61, 62,
    71, 72,
    81, 82,
}


# ==============================================================================


def calculate_metrics(y_true, y_pred, confusion_set):
    """计算总体准确率和混淆集准确率"""
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
    """绘制并保存混淆矩阵"""
    # 过滤掉背景类(48)以便看得更清楚，或者保留全部
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


@torch.no_grad()
def generate_per_graph_predictions(model, test_data_list, device, n_classes, save_path):
    """生成逐图(per-graph)预测文件"""
    logging.info(f"--- 正在生成逐图预测文件... ---")

    loader = DataLoader(test_data_list, batch_size=1, shuffle=False)
    model.eval()
    all_graph_results = []

    for batch in loader:
        batch = batch.to(device)

        # 获取 img_id
        if not hasattr(batch, 'img_id'):
            img_id = -1
        else:
            if batch.img_id.dim() > 0:
                img_id = batch.img_id[0].cpu().numpy().item()
            else:
                img_id = batch.img_id.cpu().numpy().item()

        img_path = batch.img_path[0] if isinstance(batch.img_path, list) else batch.img_path

        # 1. 运行 RecurrentAnatomyGATNew
        # forward 会执行 num_iterations 次循环，返回最终的 logits
        out_gnn = model(batch, return_att=True)

        if isinstance(out_gnn, tuple):
            # logging.info(f"进入分支A")
            logits_list, att_weights, edge_index, edge_type = out_gnn
            final_logits = logits_list[-1]
        else:
            # logging.info(f"进入分支B")
            logging.info(type(out_gnn))
            final_logits = out_gnn[-1] if isinstance(out_gnn, list) else out_gnn
            att_weights, edge_index, edge_type = None, None, None

        # pred_gnn = final_logits.argmax(dim=1)
        pred_gnn = enforce_one_to_one_matching(final_logits, background_idx=48, score_threshold=0.1)
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        if hasattr(batch, 'pred_bboxes_raw'):
            # [N, 4]
            raw_boxes = batch.pred_bboxes_raw.cpu().numpy()
        else:
            # 如果万一没有 raw，再退化到用 pos (归一化坐标)
            # 但你定义里有，所以这里基本不会走
            raw_boxes = batch.pos.cpu().numpy()

        graph_result_dict = {
            'img_id': img_id,
            'img_path': img_path,  # [新增] 直接存路径，方便读取
            'y_true': y_true.cpu(),
            'pred_original': pred_original.cpu(),
            'pred_gnn': pred_gnn.cpu(),

            # --- 可视化核心数据 ---
            'edge_index': edge_index.cpu().numpy() if edge_index is not None else None,
            'att_weights': att_weights.cpu().numpy() if att_weights is not None else None,
            'edge_types': edge_type.cpu().numpy() if edge_type is not None else None,

            'raw_boxes': raw_boxes  # 存这个！画图时算中心点: (x1+x2)/2
        }
        all_graph_results.append(graph_result_dict)

    try:
        torch.save(all_graph_results, save_path)
        logging.info(f"逐图预测文件已保存到: {save_path}")
    except Exception as e:
        logging.error(f"保存出错: {e}")


@torch.no_grad()
def evaluate_aggregate(model, loader, device, n_classes):
    """计算整个测试集的聚合指标，并测量推理速度"""
    model.eval()
    all_y_true = []
    all_pred_original = []
    all_pred_gnn = []

    # --- [新增] 速度测量变量 ---
    total_inference_time = 0.0
    total_graphs = 0

    # --- [新增] GPU 预热 (Warmup) ---
    # 跑一个小批次让 GPU 进入状态，避免第一次推理过慢影响统计
    if device.type == 'cuda':
        dummy_batch = next(iter(loader)).to(device)
        _ = model(dummy_batch)
        torch.cuda.synchronize()

    logging.info("开始批量评估与测速...")

    for batch in loader:
        batch = batch.to(device)
        num_graphs_in_batch = batch.num_graphs

        # ==========================
        # [核心] 精准计时开始
        # ==========================
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        # 模型推理
        out_gnn = model(batch)

        # [替换原有的解包逻辑]: 处理包含注意力的元组输出
        all_step_logits = out_gnn[0] if isinstance(out_gnn, tuple) else out_gnn
        final_logits = all_step_logits[-1] if isinstance(all_step_logits, list) else all_step_logits

        # 这里保持你的精准测速结束 (因为后处理不计入模型网络延迟)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        # ==========================
        # [核心] 精准计时结束
        # ==========================

        batch_time = end_time - start_time
        total_inference_time += batch_time
        total_graphs += num_graphs_in_batch

        # ------------------------------------------------------------------
        # [核心修改区开始]：单图切片应用匈牙利算法
        # ------------------------------------------------------------------
        # 预先分配一个张量存储校正后的结果
        pred_gnn_tensor = torch.zeros(final_logits.size(0), dtype=torch.long, device=device)
        node_offset = 0

        # 将 Batch 重新拆分为单图，逐图执行 1对1 约束匹配
        for graph in batch.to_data_list():
            num_nodes = graph.num_nodes
            single_img_logits = final_logits[node_offset: node_offset + num_nodes]

            # 执行匈牙利算法
            pred_gnn_tensor[node_offset: node_offset + num_nodes] = enforce_one_to_one_matching(
                single_img_logits, background_idx=48, score_threshold=0.1
            )
            node_offset += num_nodes

        pred_gnn = pred_gnn_tensor.cpu().numpy()
        # ------------------------------------------------------------------
        # [核心修改区结束]
        # ------------------------------------------------------------------
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        all_y_true.append(y_true.cpu().numpy())
        all_pred_original.append(pred_original.cpu().numpy())
        all_pred_gnn.append(pred_gnn)

    # 拼接
    y_true = np.concatenate(all_y_true)
    pred_original = np.concatenate(all_pred_original)
    pred_gnn = np.concatenate(all_pred_gnn)

    # --- [新增] 计算速度指标 ---
    avg_latency_ms = (total_inference_time / total_graphs) * 1000  # 毫秒/张
    fps = total_graphs / total_inference_time  # 张/秒

    logging.info(f"\n⚡️ 推理速度统计 ⚡️")
    logging.info(f"总耗时: {total_inference_time:.4f}s | 处理图数: {total_graphs}")
    logging.info(f"平均延迟 (Latency): {avg_latency_ms:.4f} ms/image")
    logging.info(f"吞吐量 (Throughput): {fps:.2f} FPS")

    # 返回增加速度指标
    return y_true, pred_original, pred_gnn, fps, avg_latency_ms


def main():
    # 日志设置
    log_file_path = os.path.join(RESULTS_SAVE_DIR, 'evaluation_recurrent.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"评估模型: RecurrentAnatomyGATNew (Relations={NUM_RELATIONS}, Iterations={NUM_ITERATIONS})")
    logging.info(f"测试数据文件:{TEST_DATA_PATH}")
    # --- 1. 加载 Recurrent 模型 ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"模型文件未找到: {MODEL_PATH}")
        return

    # [!! 关键 !!] 实例化时传入 num_iterations
    model = RecurrentAnatomyGATNew(
        n_classes=N_CLASSES,
        num_relations=NUM_RELATIONS,
        num_iterations=NUM_ITERATIONS
    ).to(DEVICE)
    # model = RecurrentAnatomyGATNew_A(n_classes=N_CLASSES,num_relations=4,num_iterations=1,use_prior = True,use_edge_features=False,spatial_only=True).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logging.info(f"模型权重已加载: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"加载权重失败: {e}")
        return

    # --- 2. 加载数据 ---
    try:
        test_data_list = torch.load(TEST_DATA_PATH, map_location='cpu', weights_only=False)
        logging.info(f"测试数据已加载: {len(test_data_list)} 张图")
    except Exception as e:
        logging.error(f"数据加载失败: {e}");
        return

    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 执行评估 ---
    # --- 3. 执行批量评估 ---
    # [!! 修改 !!] 接收 5 个返回值
    y_true, pred_original, pred_gnn, fps, latency = evaluate_aggregate(
        model, test_loader, DEVICE, N_CLASSES
    )

    # --- 4. 计算指标 ---
    logging.info("\n--- 计算核心指标 ---")
    metrics_orig = calculate_metrics(y_true, pred_original, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true, pred_gnn, CONFUSION_SET_CLASSES)

    logging.info("\n" + "=" * 60)
    logging.info(f"{'指标':<20} | {'Baseline':<15} | {'AnatomyGAT':<15}")
    logging.info("-" * 60)
    logging.info(f"{'总体准确率':<20} | {metrics_orig[0]:.4f}          | {metrics_gnn[0]:.4f}")
    logging.info(f"{'混淆集准确率':<20} | {metrics_orig[1]:.4f}          | {metrics_gnn[1]:.4f}")
    # [!! 新增 !!] 打印速度
    logging.info("-" * 60)
    logging.info(f"{'推理速度 (FPS)':<20} | {'-':<15} | {fps:.2f}")
    logging.info(f"{'单张延迟 (ms)':<20} | {'-':<15} | {latency:.2f}")
    logging.info("=" * 60 + "\n")

    # --- 5. 生成详细报告 ---
    simple_class_names = [str(i) for i in range(N_CLASSES)]

    report_gnn = classification_report(y_true, pred_gnn, target_names=simple_class_names, zero_division=0)
    logging.info("RecurrentGAT 分类报告预览:\n" + report_gnn)

    with open(os.path.join(RESULTS_SAVE_DIR, 'classification_report_recurrent.txt'), 'w') as f:
        f.write(report_gnn)

    # --- 6. 绘制混淆矩阵 ---
    plot_confusion_matrix(
        y_true, pred_gnn, simple_class_names,
        f'Confusion Matrix - RecurrentGAT (T={NUM_ITERATIONS})',
        os.path.join(RESULTS_SAVE_DIR, 'cm_recurrent_gat.png')
    )

    # --- 7. 生成逐图预测文件 ---
    save_pt_path = os.path.join(RESULTS_SAVE_DIR, 'graph_by_graph_predictions_recurrent_ccsw_60data_i5_r2.pt')
    generate_per_graph_predictions(model, test_data_list, DEVICE, N_CLASSES, save_pt_path)

    logging.info("迭代模型评估结束。")


if __name__ == '__main__':
    main()