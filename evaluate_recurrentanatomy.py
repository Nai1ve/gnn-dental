import logging
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import RecurrentAnatomyGAT

# ==============================================================================
# 1. 核心配置
# ==============================================================================

MODEL_PATH = 'checkpoints/2025-11-20_20-28-43/best_model.pth'

TEST_DATA_PATH = 'gnn_data/test_ccsw.pt'  # 请确保这是包含 V4.1+ 结构的新数据
N_CLASSES = 49
NUM_RELATIONS = 4
NUM_ITERATIONS = 3  # [!! 新增 !!] 必须与训练时设置的迭代次数一致
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

        # 1. 运行 RecurrentAnatomyGAT
        # forward 会执行 num_iterations 次循环，返回最终的 logits
        out_gnn = model(batch)
        if isinstance(out_gnn, list):
            final_logits = out_gnn[-1]
        else:
            final_logits = out_gnn
        pred_gnn = final_logits.argmax(dim=1)
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        graph_result_dict = {
            'img_id': img_id,
            'y_true': y_true.cpu(),
            'pred_original': pred_original.cpu(),
            'pred_gnn': pred_gnn.cpu()
        }
        all_graph_results.append(graph_result_dict)

    try:
        torch.save(all_graph_results, save_path)
        logging.info(f"逐图预测文件已保存到: {save_path}")
    except Exception as e:
        logging.error(f"保存出错: {e}")


@torch.no_grad()
def evaluate_aggregate(model, loader, device, n_classes):
    """计算整个测试集的聚合结果"""
    model.eval()
    all_y_true = []
    all_pred_original = []
    all_pred_gnn = []

    logging.info("开始批量评估...")
    for batch in loader:
        batch = batch.to(device)

        out_gnn = model(batch)
        if isinstance(out_gnn, list):
            final_logits = out_gnn[-1]
        else:
            final_logits = out_gnn
        pred_gnn = final_logits.argmax(dim=1)

        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        all_y_true.append(y_true.cpu().numpy())
        all_pred_original.append(pred_original.cpu().numpy())
        all_pred_gnn.append(pred_gnn.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    pred_original = np.concatenate(all_pred_original)
    pred_gnn = np.concatenate(all_pred_gnn)

    return y_true, pred_original, pred_gnn


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
    logging.info(f"评估模型: RecurrentAnatomyGAT (Relations={NUM_RELATIONS}, Iterations={NUM_ITERATIONS})")

    # --- 1. 加载 Recurrent 模型 ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"模型文件未找到: {MODEL_PATH}")
        return

    # [!! 关键 !!] 实例化时传入 num_iterations
    model = RecurrentAnatomyGAT(
        n_classes=N_CLASSES,
        num_relations=NUM_RELATIONS,
        num_iterations=NUM_ITERATIONS
    ).to(DEVICE)

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
    y_true, pred_original, pred_gnn = evaluate_aggregate(model, test_loader, DEVICE, N_CLASSES)

    # --- 4. 计算指标 ---
    logging.info("\n--- 计算核心指标 ---")
    metrics_orig = calculate_metrics(y_true, pred_original, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true, pred_gnn, CONFUSION_SET_CLASSES)

    logging.info("\n" + "=" * 65)
    logging.info(f"{'指标':<20} | {'Baseline (Det)':<15} | {'RecurrentGAT (Ours)':<20}")
    logging.info("-" * 65)
    logging.info(f"{'总体准确率':<20} | {metrics_orig[0]:.4f}          | {metrics_gnn[0]:.4f}")
    logging.info(f"{'混淆集准确率':<20} | {metrics_orig[1]:.4f}          | {metrics_gnn[1]:.4f}")
    logging.info("=" * 65 + "\n")

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
    save_pt_path = os.path.join(RESULTS_SAVE_DIR, 'graph_by_graph_predictions_recurrent_ccsw_final.pt')
    generate_per_graph_predictions(model, test_data_list, DEVICE, N_CLASSES, save_pt_path)

    logging.info("迭代模型评估结束。")


if __name__ == '__main__':
    main()