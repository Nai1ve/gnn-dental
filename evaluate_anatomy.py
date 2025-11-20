import logging
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import AnatomyGAT


# ==============================================================================
# 1. 核心配置
# ==============================================================================
MODEL_PATH = 'checkpoints/2025-11-19_10-27-34/best_model_confusion.pth'

TEST_DATA_PATH = 'gnn_data/test_v4.pt'  # 测试集路径
N_CLASSES = 49  # 类别数
NUM_RELATIONS = 3  # [!! 新增 !!] 关系数 (Overlap, Arch, Spatial)
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 结果保存目录 (默认为模型所在目录)
RESULTS_SAVE_DIR = os.path.dirname(MODEL_PATH)

# [!! 核心 !!] 混淆集定义 (必须与 train.py 一致)
CONFUSION_SET_CLASSES = {
    11, 12,  # 上颌右侧切牙
    21, 22,  # 上颌左侧切牙
    31, 32,  # 下颌左侧切牙
    41, 42  # 下颌右侧切牙
}


# ==============================================================================


def calculate_metrics(y_true, y_pred, confusion_set):
    """计算总体准确率和混淆集准确率"""
    # 1. 总体准确率
    overall_acc = accuracy_score(y_true, y_pred)

    # 2. 混淆集准确率
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
    # 为了清晰，只绘制前48个类别（忽略背景，或者根据需要调整）
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

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
    """
    生成"逐图"预测文件，保存为 list of dicts。
    用于外部可视化和详细分析。
    """
    logging.info(f"--- 正在生成逐图(per-graph)预测文件... ---")

    # 使用 batch_size=1 逐图处理
    loader = DataLoader(test_data_list, batch_size=1, shuffle=False)
    model.eval()
    all_graph_results = []

    for batch in loader:
        batch = batch.to(device)

        # --- 0. 获取 img_id (健壮性检查) ---
        if not hasattr(batch, 'img_id'):
            logging.warning("数据缺少 img_id 属性，将使用 -1 代替。")
            img_id = -1
        else:
            if batch.img_id.dim() > 0:
                img_id = batch.img_id[0].cpu().numpy().item()
            else:
                img_id = batch.img_id.cpu().numpy().item()

        # --- 1. 运行 AnatomyGAT ---
        out_gnn = model(batch)

        # --- 2. 获取结果 ---
        pred_gnn = out_gnn.argmax(dim=1)
        # 假设 x_prior 的前 49 维是 softmax
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        # --- 3. 创建结果字典 ---
        graph_result_dict = {
            'img_id': img_id,  # 原始图像 ID
            'y_true': y_true.cpu(),  # 真值
            'pred_original': pred_original.cpu(),  # Baseline 预测
            'pred_gnn': pred_gnn.cpu()  # AnatomyGAT 预测
        }
        all_graph_results.append(graph_result_dict)

    # --- 4. 保存 ---
    try:
        torch.save(all_graph_results, save_path)
        logging.info(f"逐图预测文件已保存到: {save_path}")
    except Exception as e:
        logging.error(f"保存出错: {e}")


@torch.no_grad()
def evaluate_aggregate(model, loader, device, n_classes):
    """计算整个测试集的聚合指标"""
    model.eval()
    all_y_true = []
    all_pred_original = []
    all_pred_gnn = []

    logging.info("开始批量评估...")
    for batch in loader:
        batch = batch.to(device)

        # AnatomyGAT 推理
        out_gnn = model(batch)
        pred_gnn = out_gnn.argmax(dim=1)

        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        y_true = batch.y.long()

        all_y_true.append(y_true.cpu().numpy())
        all_pred_original.append(pred_original.cpu().numpy())
        all_pred_gnn.append(pred_gnn.cpu().numpy())

    # 拼接
    y_true = np.concatenate(all_y_true)
    pred_original = np.concatenate(all_pred_original)
    pred_gnn = np.concatenate(all_pred_gnn)

    return y_true, pred_original, pred_gnn


def main():
    # 设置日志
    log_file_path = os.path.join(RESULTS_SAVE_DIR, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"评估 AnatomyGAT (Relations={NUM_RELATIONS})")

    # --- 1. 加载模型 ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"模型文件未找到: {MODEL_PATH}")
        return

    # [!! 修改 !!] 实例化 AnatomyGAT
    model = AnatomyGAT(n_classes=N_CLASSES, num_relations=NUM_RELATIONS).to(DEVICE)

    # 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logging.info(f"模型权重已加载: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"加载权重失败 (请检查模型结构是否匹配): {e}")
        return

    # --- 2. 加载数据 ---
    try:
        test_data_list = torch.load(TEST_DATA_PATH, map_location='cpu', weights_only=False)
        logging.info(f"测试数据已加载: {len(test_data_list)} 张图")
    except Exception as e:
        logging.error(f"数据加载失败: {e}");
        return

    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 执行批量评估 ---
    y_true, pred_original, pred_gnn = evaluate_aggregate(model, test_loader, DEVICE, N_CLASSES)

    # --- 4. 计算指标 ---
    logging.info("\n--- 计算核心指标 ---")
    metrics_orig = calculate_metrics(y_true, pred_original, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true, pred_gnn, CONFUSION_SET_CLASSES)

    logging.info("\n" + "=" * 60)
    logging.info(f"{'指标':<20} | {'Baseline (Det)':<15} | {'AnatomyGAT (Ours)':<15}")
    logging.info("-" * 60)
    logging.info(f"{'总体准确率':<20} | {metrics_orig[0]:.4f}          | {metrics_gnn[0]:.4f}")
    logging.info(f"{'混淆集准确率':<20} | {metrics_orig[1]:.4f}          | {metrics_gnn[1]:.4f}")
    logging.info("=" * 60 + "\n")

    # --- 5. 生成详细报告 ---
    simple_class_names = [str(i) for i in range(N_CLASSES)]  # 0-48

    report_gnn = classification_report(y_true, pred_gnn, target_names=simple_class_names, zero_division=0)
    logging.info("AnatomyGAT 分类报告预览:\n" + report_gnn)

    with open(os.path.join(RESULTS_SAVE_DIR, 'classification_report_anatomy.txt'), 'w') as f:
        f.write(report_gnn)

    # --- 6. 绘制混淆矩阵 ---
    plot_confusion_matrix(
        y_true, pred_gnn, simple_class_names,
        'Confusion Matrix - AnatomyGAT',
        os.path.join(RESULTS_SAVE_DIR, 'cm_anatomy_gat.png')
    )

    # --- 7. [关键] 生成逐图预测文件 ---
    save_pt_path = os.path.join(RESULTS_SAVE_DIR, 'graph_by_graph_predictions.pt')
    generate_per_graph_predictions(model, test_data_list, DEVICE, N_CLASSES, save_pt_path)

    logging.info("评估流程结束。")


if __name__ == '__main__':
    main()