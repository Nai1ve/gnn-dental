from collections import defaultdict

import torch
from torch_geometric.data import DataLoader

from focal_loss import FocalLoss
from model import GAT_Numbering_Corrector, GAT_Numbering_Corrector_V2
from util import calculate_weights
import logging
import os
import pickle

# --- 配置 ---
BEST_MODEL_PATH = 'checkpoints/best_model.pth'
TEST_DATA_PATH = 'gnn_test_data.pt'

OUTPUT_RESULTS_PATH = 'gnn_corrected_results.pkl'

BATCH_SIZE = 128
INPUT_CHANNELS = 55
INPUT_CHANNELS_V3 = 1030
HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = 49
HEADS = 4
DROPOUT_RATE = 0.2 # 如果你用了V2模型，确保这个值匹配
BACKGROUND_CLASS_INDEX = 48

USE_V2_MODEL = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test_model(model,loader,criterion):
    """测试集上评估"""
    model.eval()
    total_loss = 0
    correct_nodes = 0
    total_nodes = 0

    results_by_image = defaultdict(dict)

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        gnn_preds = out.argmax(dim=1)

        loss = criterion(out,batch.y)
        total_loss += loss.item() * batch.num_graphs

        pred = out.argmax(dim=1)
        # 节点分类准确率
        correct_nodes += int((pred == batch.y).sum())
        total_nodes += batch.num_nodes

        # 解包数据
        graph_list = batch.to_data_list()
        node_offset = 0
        for graph in graph_list:
            num_nodes_in_graph = graph.num_nodes

            img_id = graph.img_id

            # 从整个预测结果中切出书序当前图的部分
            node_predictions = gnn_preds[node_offset : node_offset + num_nodes_in_graph]

            image_predictions = []

            for i in range(num_nodes_in_graph):
                is_background = (node_predictions[i].item() == BACKGROUND_CLASS_INDEX)

                # 我们只保留GNN认为不是背景的预测
                if not is_background:
                    prediction_details = {
                        # 原始Faster R-CNN的信息
                        'bbox': graph.pred_bboxes_raw[i].cpu().numpy(),
                        'original_score': graph.pred_scores_raw[i].item(),
                        'original_label_idx': graph.pred_labels_raw[i].item(),
                        # GNN修正后的信息
                        'corrected_label_idx': node_predictions[i].item()
                    }
                    image_predictions.append(prediction_details)

            # 将这张图片的所有有效预测存入我们的主字典
            results_by_image[img_id] = {
                'img_path': graph.img_path,
                'ori_shape': graph.ori_shape.cpu().numpy(),
                'predictions': image_predictions
            }

            # 更新偏移量，处理下一张图
            node_offset += num_nodes_in_graph

    avg_loss = total_loss / len(loader.dataset)
    test_accuracy = correct_nodes / total_nodes

    return results_by_image,avg_loss,test_accuracy


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"使用设备: {device}")

    # --- 1. 加载测试数据 ---
    if not os.path.exists(TEST_DATA_PATH):
        logging.error(f"测试数据文件未找到: {TEST_DATA_PATH}")
        return

    logging.info(f"正在加载测试数据从: {TEST_DATA_PATH}")
    test_data_list = torch.load(TEST_DATA_PATH)
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. 初始化模型 ---
    logging.info(f"正在初始化模型...")
    if USE_V2_MODEL:
        model = GAT_Numbering_Corrector_V2(
            in_channels=INPUT_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            heads=HEADS,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        logging.info("使用了 V2 模型 (3层, MLP头)")
    else:
        model = GAT_Numbering_Corrector(
            in_channels=INPUT_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            heads=HEADS
        ).to(device)
        logging.info("使用了 V1 模型 (2层)")

    # --- 3. 加载训练好的模型权重 ---
    if not os.path.exists(BEST_MODEL_PATH):
        logging.error(f"最佳模型文件未找到: {BEST_MODEL_PATH}")
        return

    logging.info(f"正在加载模型权重从: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # --- 4. 定义损失函数 (用于计算loss) ---
    if os.path.exists('gnn_train_data_0.05.pt'):
        train_data_list = torch.load('gnn_train_data_0.05.pt')
        alpha_weights = calculate_weights(train_data_list).to(device)
    else:
        alpha_weights = torch.ones(OUTPUT_CHANNELS).to(device)  # Fallback
    criterion = FocalLoss(alpha_weights, gamma=2)

    # --- 5. 执行测试 ---
    logging.info("开始在测试集上进行最终评估...")
    final_results,test_loss, test_accuracy = test_model(model, test_loader, criterion)

    logging.info("\n" + "=" * 30)
    logging.info("--- 测试评估结果 ---")
    logging.info(f"测试集损失 (Test Loss): {test_loss:.4f}")
    logging.info(f"测试集节点准确率 (Test Accuracy): {test_accuracy:.4f} (或 {test_accuracy:.2%})")
    logging.info("=" * 30)

    # --- 6. 保存预测结果以供详细分析 ---
    # 这个文件可以被你的错误分析脚本读取，来计算最终的FP-Class, FP-Hallu等指标
    # --- 5. 将最终的结构化结果保存到文件 ---
    logging.info(f"正在将 {len(final_results)} 张图片的修正结果保存到: {OUTPUT_RESULTS_PATH}")
    with open(OUTPUT_RESULTS_PATH, 'wb') as f:
        pickle.dump(final_results, f)
    logging.info("保存完成。")


if __name__ == '__main__':
    main()