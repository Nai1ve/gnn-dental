import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    """
    多分类场景下的Focal Loss实现
    """

    def __init__(self,alpha=None,gamma=2.0,reduction='mean'):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction


    def forward(self,input,targets):
        ce_loss = F.cross_entropy(input,targets,reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha,float):
                alpha_t = torch.full_like(targets,1-self.alpha)
                alpha_t[targets !=0] = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



if __name__ == '__main__':
    # 1. 实例化两种损失函数
    # 注意：我们将reduction设置为'none'，以便观察每个样本的损失
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    focal_loss_fn = FocalLoss(gamma=2, alpha=None, reduction='none')  # alpha暂时不用，专注于gamma的效果

    # 2. 创建模拟数据
    # 假设我们有4个样本，10个类别 (类别0是多数背景类)
    # logits是模型未经softmax的输出
    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 0, 3, 3])  # 两个背景样本，两个前景样本

    # --- 精心设计logits来模拟不同难度的样本 ---

    # 样本0: 简单负样本 (Easy Negative)
    # 真实标签是0, 模型以极高置信度预测为0
    logits[0, 0] = 10.0  # 极高的分数给到正确类别0
    logits[0, 1:] = 1.0  # 其他类别的分数很低

    # 样本1: 困难负样本 (Hard Negative)
    # 真实标签是0, 但模型错误地认为它可能是类别3
    logits[1, 0] = 1.5  # 正确类别0的分数不高
    logits[1, 3] = 1.0  # 其他错误类别的分数与之接近
    logits[1, [1, 2, 4, 5, 6, 7, 8, 9]] = 0.1

    # 样本2: 简单正样本 (Easy Positive)
    # 真实标签是3, 模型以较高置信度预测为3
    logits[2, 3] = 5.0  # 较高分数给到正确类别3
    logits[2, [0, 1, 2, 4, 5, 6, 7, 8, 9]] = 0.5

    # 样本3: 困难正样本 (Hard Positive)
    # 真实标签是3, 但模型非常不确定，甚至认为它更可能是0
    logits[3, 3] = 1.0  # 正确类别3的分数很低
    logits[3, 0] = 1.2  # 最高的错误分数给了类别0

    # 3. 计算并打印结果
    print("--- Focal Loss 效果验证 ---")
    probs = F.softmax(logits, dim=1)
    ce_losses = ce_loss_fn(logits, labels)
    focal_losses = focal_loss_fn(logits, labels)


    # 格式化打印结果
    def print_results(idx, case_name):
        print(f"\n--- 案例 {idx}: {case_name} ---")
        print(f"真实标签 (Label)      : {labels[idx].item()}")
        print(f"预测概率 (Probabilities): [类别0: {probs[idx, 0]:.4f}, 类别3: {probs[idx, 3]:.4f}]")
        print(f"预测正确类别的概率(pt): {probs[idx, labels[idx]].item():.4f}")
        print(f"标准交叉熵损失 (CE)  : {ce_losses[idx].item():.4f}")
        print(f"Focal Loss (FL)      : {focal_losses[idx].item():.4f}")
        if ce_losses[idx].item() > 1e-5:
            ratio = focal_losses[idx].item() / ce_losses[idx].item()
            print(f"FL / CE 比例         : {ratio:.4f} (损失被抑制为原来的 {ratio * 100:.2f}%)")


    print_results(0, "简单负样本 (Easy Negative)")
    print_results(1, "困难负样本 (Hard Negative)")
    print_results(2, "简单正样本 (Easy Positive)")
    print_results(3, "困难正样本 (Hard Positive)")