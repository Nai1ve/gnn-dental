import logging
from typing import List

from torch import Tensor
from torch_geometric.data import Data
import torch


NUM_CLASSES = 49

def calculate_weights(train_data_list:List[Data])->Tensor:
    """
    计算Focal Loss的权重

    :param train_data_list:
    :return:
    """
    logging.info("正在根据训练集数据计算类别权重...")
    if not train_data_list:
        logging.warning("训练数据列表为空，无法计算权重。")
        return torch.ones(NUM_CLASSES) # 返回默认权重

    all_labels = [data.y for data in train_data_list]

    class_counts = torch.bincount(all_labels, minlength=NUM_CLASSES)
    logging.info(f"类别频率统计 (0-{NUM_CLASSES-1}): {class_counts.tolist()}")


    weights = 1.0 / (class_counts.float() + 1e-6)
    normalized_weights = weights / weights.max()

    logging.info(f"权重计算完成。背景类(索引 {NUM_CLASSES-1}) 的归一化权重: {normalized_weights[-1].item():.4f}")


    return normalized_weights