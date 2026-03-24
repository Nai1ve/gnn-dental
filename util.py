import logging
from typing import List

from torch import Tensor
from torch_geometric.data import Data
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 49


def calculate_weights(train_data_list: List[Data]) -> Tensor:
    """
    计算Focal Loss的权重

    :param train_data_list:
    :return:
    """
    logging.info("正在根据训练集数据计算类别权重...")
    if not train_data_list:
        logging.warning("训练数据列表为空，无法计算权重。")
        return torch.ones(NUM_CLASSES)  # 返回默认权重

    all_labels = torch.cat([data.y for data in train_data_list])

    class_counts = torch.bincount(all_labels, minlength=NUM_CLASSES)
    logging.info(f"类别频率统计 (0-{NUM_CLASSES - 1}): {class_counts.tolist()}")

    weights = 1.0 / (class_counts.float() + 1e-6)
    normalized_weights = weights / weights.max()

    logging.info(f"权重计算完成。背景类(索引 {NUM_CLASSES - 1}) 的归一化权重: {normalized_weights[-1].item():.4f}")

    return normalized_weights


def enforce_one_to_one_matching(logits, background_idx=48, score_threshold=0.3):
    """
    使用二分图匹配 (匈牙利算法) 实现一对一的临床解剖输出。

    参数:
        logits: GNN 最后一步的输出张量，形状 [N, 49]
        background_idx: 背景类的索引 (默认 48)
        score_threshold: 拒绝分配的最低置信度阈值 (低于此阈值强行归为背景)
    返回:
        一维 Tensor，形状 [N]，包含了绝对一对一的类别预测结果
    """
    device = logits.device

    # 1. 转换为概率分布并转到 CPU 计算
    probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
    N = probs.shape[0]

    # 默认所有框初始化为背景类
    final_preds = np.full(N, background_idx, dtype=np.int64)

    if N == 0:
        return torch.tensor(final_preds, device=device)


    cost_matrix = -probs[:, :background_idx]  # 形状: [N, 48]


    row_ind, col_ind = linear_sum_assignment(cost_matrix)


    for r, c in zip(row_ind, col_ind):

        if probs[r, background_idx] > probs[r, c] or probs[r, c] < score_threshold:
            continue
        final_preds[r] = c

    return torch.tensor(final_preds, device=device)

