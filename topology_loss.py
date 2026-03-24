import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalConsistencyLoss(nn.Module):
    def __init__(self, class_names, device):
        super(TopologicalConsistencyLoss, self).__init__()
        self.device = device
        self.n_classes = len(class_names)  # 48
        # 构建 FDI 牙齿非法邻接惩罚矩阵 (48 x 48)
        self.penalty_matrix = self._build_fdi_penalty_matrix(class_names).to(device)

    def _build_fdi_penalty_matrix(self, class_names):
        """
        为混合牙列设计的宽容惩罚矩阵。
        默认合法为 0，非法为 1。
        """
        # 初始化全 0 (默认都不惩罚，包容乳恒牙交替的复杂相邻情况)
        mat = torch.zeros((self.n_classes, self.n_classes), dtype=torch.float32)

        # 非法 1：在牙弓相邻的两个节点，绝不可能预测成【同一颗牙】 (FP-C 同化混淆)
        mat.fill_diagonal_(1.0)

        # 非法 2：左右半口/上下颌的后牙绝对不可能相邻 (跨象限惩罚)
        # 例如 36 (左下) 旁边绝对不可能是 46 (右下) 或者 26 (左上)
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        for i, name_a in enumerate(class_names):
            for j, name_b in enumerate(class_names):
                # 如果同侧 (比如都是左下 '3')，不惩罚
                if name_a[0] == name_b[0]:
                    continue
                # 如果都是切牙 (尾号是 1 或 2，比如 11和21, 32和41)，允许跨象限相邻，不惩罚
                if name_a[1] in ['1', '2'] and name_b[1] in ['1', '2']:
                    continue
                # 剩下的情况 (比如左边磨牙和右边磨牙)，设为非法惩罚
                mat[i, j] = 1.0

        return mat

    def forward(self, logits, edge_index, edge_type, attention_weights):
        """
        参数:
            logits: 模型输出 [N, 49]
            edge_index: 全局拼合的边索引 [2, E]
            edge_type: 边类型 [E] (0: overlap, 1: arch, ...)
            attention_weights: GNN 预测的动态边权重 [E]
        """
        probs = F.softmax(logits, dim=-1)
        probs_tooth = probs[:, :self.n_classes]  # [N, 48]

        # 提取各个类型的掩码
        mask_overlap = (edge_type == 0)
        mask_arch = (edge_type == 1)

        # 1. 动态 DUPS Loss (注意力调制的正交惩罚)
        loss_dups = torch.tensor(0.0, device=self.device)
        if mask_overlap.sum() > 0:
            src_ov = edge_index[0, mask_overlap]
            dst_ov = edge_index[1, mask_overlap]
            alpha_ov = attention_weights[mask_overlap]  # 获取重叠边的动态权重

            overlap_dot = torch.sum(probs_tooth[src_ov] * probs_tooth[dst_ov], dim=-1)
            # 用网络自身学到的注意力来加权惩罚
            loss_dups = torch.sum(overlap_dot * alpha_ov) / (alpha_ov.sum() + 1e-6)

        # 2. 动态 FP-C Loss (注意力调制的非法转移惩罚)
        loss_fpc = torch.tensor(0.0, device=self.device)
        if mask_arch.sum() > 0:
            src_ar = edge_index[0, mask_arch]
            dst_ar = edge_index[1, mask_arch]
            alpha_ar = attention_weights[mask_arch]  # 获取牙弓边的动态权重

            src_penalty = torch.matmul(probs_tooth[src_ar], self.penalty_matrix)
            transition_penalty = torch.sum(src_penalty * probs_tooth[dst_ar], dim=-1)
            # 网络认为不是真边的，惩罚力度自动降低
            loss_fpc = torch.sum(transition_penalty * alpha_ar) / (alpha_ar.sum() + 1e-6)

        return loss_dups, loss_fpc