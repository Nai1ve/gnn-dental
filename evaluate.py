import logging
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import BaselineGNN


# --- 核心配置 ---
MODEL_PATH = 'checkpoints/2025-11-18_12-40-36/best_model.pth'
TEST_DATA_PATH = 'gnn_data/test.pt'
RESULTS_SAVE_DIR = os.path.dirname(MODEL_PATH)

# --- 确保配置与训练时一致 ---
N_CLASSES = 49
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [!! 核心 !!] 必须与 train.py 中的定义完全一致
CONFUSION_SET_CLASSES = {
    11, 12,  # 上颌右侧切牙
    21, 22,  # 上颌左侧切牙
    31, 32,  # 下颌左侧切牙
    41, 42  # 下颌右侧切牙
}


# --- 结束配置 ---


def calculate_metrics(y_true, y_pred, confusion_set):
    """
    计算总体准确率和混淆集准确率。
    这是 `train.py` 中那个bug的“Numpy修正版”。
    """
    # 1. 总体准确率 (Overall Accuracy)
    overall_acc = accuracy_score(y_true, y_pred)

    # 2. 混淆集准确率 (Confusion-Set Accuracy)
    # [!! 核心修正 !!]
    # 使用 np.isin 来高效地创建 mask
    mask = np.isin(y_true, list(confusion_set))

    total_confusion_nodes = mask.sum()
    if total_confusion_nodes == 0:
        logging.warning(f"测试集中未找到任何属于 {confusion_set} 的节点。")
        confusion_set_acc = 0.0
    else:
        # 比较 y_true[mask] 和 y_pred[mask]
        correct_confusion_nodes = (y_true[mask] == y_pred[mask]).sum()
        confusion_set_acc = correct_confusion_nodes / total_confusion_nodes

    return overall_acc, confusion_set_acc


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """绘制并保存混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)

    # 为了可读性，我们可能只想绘制“混淆集”的子矩阵
    # (这里为了完整性，我们先绘制完整的)

    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"混淆矩阵已保存到: {save_path}")
    plt.close()


@torch.no_grad()
def evaluate(model, loader, device, confusion_set, n_classes):
    model.eval()

    all_y_true = []
    all_pred_original = []
    all_pred_gnn = []
    all_graph_indices = []

    logging.info("开始在测试集上评估...")
    graph_counter = 0
    for batch in loader:
        batch = batch.to(device)

        # --- 1. 获取 GNN 预测 (修正后) ---
        out_gnn = model(batch)
        pred_gnn = out_gnn.argmax(dim=1)

        # --- 2. 获取原始检测器预测 (修正前) ---
        # 我们假设 x_prior 的前 N_CLASSES 维是 softmax
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)

        # --- 3. 获取真实标签 ---
        y_true = batch.y.long()

        global_graph_index_vector = batch.batch + graph_counter
        graph_counter += batch.num_graphs

        # 收集所有结果
        all_y_true.append(y_true.cpu())
        all_pred_original.append(pred_original.cpu())
        all_pred_gnn.append(pred_gnn.cpu())
        all_graph_indices.append(global_graph_index_vector.cpu())

    logging.info("评估完成，正在拼接结果...")

    # 将所有批次的结果拼接成一个大数组
    y_true = torch.cat(all_y_true)
    pred_original = torch.cat(all_pred_original)
    pred_gnn = torch.cat(all_pred_gnn)
    graph_indices = torch.cat(all_graph_indices)

    return y_true, pred_original, pred_gnn, graph_indices


@torch.no_grad()
def generate_per_graph_predictions(model, test_data_list, device, n_classes, save_path):
    """
    为评估脚本生成"逐图"的预测文件。
    它会创建一个list，其中每个元素是一个包含单张图所有预测的dict。
    """
    logging.info(f"--- 正在生成逐图(per-graph)预测文件 (共 {len(test_data_list)} 张图)... ---")

    # [!! 核心 !!] 我们为这个任务创建一个新的、batch_size=1 的 loader
    # 这样我们就可以逐图迭代，并获取 'i' 作为 image_id
    loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

    model.eval()
    all_graph_results = []  # 这将是我们最终保存的列表

    # 'i' 将是我们的 image_id (0, 1, 2, ...)
    for i, batch in enumerate(loader):
        batch = batch.to(device)

        # --- 1. 运行模型 ---
        out_gnn = model(batch)

        # --- 2. 获取所有预测和真值 ---
        pred_gnn = out_gnn.argmax(dim=1)
        pred_original = batch.x_prior[:, :n_classes].argmax(dim=1)
        img_id= batch.img_id.long()
        y_true = batch.y.long()

        # --- 3. 创建此图的结果字典 ---
        graph_result_dict = {
            'img_id': img_id.cpu().numpy().item(),  # 图的ID (索引)
            'y_true': y_true.cpu(),  # 真值 (Tensor)
            'pred_original': pred_original.cpu(),  # 原始预测 (Tensor)
            'pred_gnn': pred_gnn.cpu()  # GNN预测 (Tensor)
        }
        all_graph_results.append(graph_result_dict)

    # --- 4. 保存这个包含所有字典的列表 ---
    try:
        torch.save(all_graph_results, save_path)
        logging.info(f"逐图预测文件已保存到: {save_path}")
    except Exception as e:
        logging.error(f"保存逐图预测文件时出错: {e}")


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
    logging.info(f"结果将保存在: {RESULTS_SAVE_DIR}")

    # --- 1. 加载模型 ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"模型文件未找到: {MODEL_PATH}")
        logging.error("请更新脚本顶部的 'MODEL_PATH' 变量。")
        return

    model = BaselineGNN(n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    logging.info(f"模型已从 {MODEL_PATH} 加载。")

    # --- 2. 加载数据 ---
    try:
        test_data_list = torch.load(TEST_DATA_PATH, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        logging.error(f"测试数据未找到: {TEST_DATA_PATH}")
        return

    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)
    logging.info(f"测试数据: {len(test_data_list)} 个图已加载。")

    # --- 3. 执行评估 ---
    y_true, pred_original, pred_gnn, graph_indices = evaluate(
        model, test_loader, DEVICE, CONFUSION_SET_CLASSES, N_CLASSES
    )

    y_true_np = y_true.numpy()
    pred_original_np = pred_original.numpy()
    pred_gnn_np = pred_gnn.numpy()

    # --- 4. 计算和比较指标 ---
    logging.info("\n--- 正在计算指标... ---")

    metrics_original = calculate_metrics(y_true_np, pred_original_np, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true_np, pred_gnn_np, CONFUSION_SET_CLASSES)

    logging.info("\n" + "=" * 50)
    logging.info("          测试集指标对比")
    logging.info("=" * 50)
    logging.info(f"指标              | 原始检测 (Baseline) | GNN修正后 (Ours)")
    logging.info(f"------------------|-----------------------|-------------------")
    logging.info(f"总体准确率        | {metrics_original[0]:.4f}                  | {metrics_gnn[0]:.4f}")
    logging.info(f"混淆集准确率      | {metrics_original[1]:.4f}                  | {metrics_gnn[1]:.4f}")
    logging.info("=" * 50 + "\n")

    # --- 5. 生成详细报告 (Sklearn) ---
    # (假设类别0是背景，48是牙齿，所以总共49个名字)
    # TODO: 您可以修改这里的 class_names
    class_names = ['48'] + [str(i) for i in [
    '11', '12', '13', '14', '15', '16', '17',
    '21', '22', '23', '24', '25', '26', '27',
    '31', '32', '33', '34', '35', '36', '37',
    '41', '42', '43', '44', '45', '46', '47',
    '51', '52', '53', '54', '55',
    '61', '62', '63', '64', '65',
    '71', '72', '73', '74', '75',
    '81', '82', '83', '84', '85'
    ]]  # (注意：这里的顺序必须和您数据中的索引严格一致！)
    # 鉴于您的类别有49个，您可能需要一个更简单的列表
    simple_class_names = [str(i) for i in range(N_CLASSES)]

    logging.info("\n--- 原始检测 (Baseline) 分类报告 ---")
    report_original = classification_report(y_true_np, pred_original_np, target_names=simple_class_names,
                                            zero_division=0)
    logging.info(report_original)

    logging.info("\n--- GNN修正后 (Ours) 分类报告 ---")
    report_gnn = classification_report(y_true, pred_gnn, target_names=simple_class_names, zero_division=0)
    logging.info(report_gnn)

    # 将报告保存到文件
    with open(os.path.join(RESULTS_SAVE_DIR, 'classification_reports.txt'), 'w') as f:
        f.write("--- 原始检测 (Baseline) 分类报告 ---\n")
        f.write(report_original)
        f.write("\n\n--- GNN修正后 (Ours) 分类报告 ---\n")
        f.write(report_gnn)

    # --- 6. 绘制混淆矩阵 ---
    plot_confusion_matrix(
        y_true_np, pred_original_np, simple_class_names,
        'Confusion Matrix - Original Detection (Baseline)',
        os.path.join(RESULTS_SAVE_DIR, 'cm_original.png')
    )
    plot_confusion_matrix(
        y_true_np, pred_gnn_np, simple_class_names,
        'Confusion Matrix - GNN Corrected (Ours)',
        os.path.join(RESULTS_SAVE_DIR, 'cm_gnn_corrected.png')
    )

    logging.info("\n--- 正在调用函数以保存逐图(per-graph)预测文件... ---")

    # 定义您想保存的文件路径
    per_graph_save_path = os.path.join(RESULTS_SAVE_DIR, 'graph_by_graph_predictions.pt')

    # 调用我们新添加的函数
    # 它需要: model, test_data_list (不是loader!), device, n_classes, save_path
    generate_per_graph_predictions(
        model,
        test_data_list,  # <-- 注意: 传入的是列表, 不是loader
        DEVICE,
        N_CLASSES,
        per_graph_save_path
    )

    logging.info("评估脚本执行完毕。")


if __name__ == '__main__':
    main()