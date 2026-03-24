import logging
import os
import torch.optim
from model import GAT_Numbering_Corrector, GAT_Numbering_Corrector_V2, GAT_Numbering_Corrector_V3, BaselineGNN, \
    AnatomyGAT, RecurrentAnatomyGAT, RecurrentAnatomyGATNew, RecurrentAnatomyGATNew_A
from focal_loss import FocalLoss
from util import calculate_weights
from torch_geometric.loader import DataLoader
from datetime import datetime
from topology_loss import TopologicalConsistencyLoss
import matplotlib.pyplot as plt
from util import enforce_one_to_one_matching

USE_EDGE_FEATURES = True  # 消融 1: 是否启用动态边缘特征
USE_TOPO_LOSS = True  # 消融 2: 是否启用 RATM 拓扑损失 (L_topo)
USE_ASYMMETRIC_LOSS = True  # 消融 3: 是否启用非对称 Focal Loss
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_path = 'data/gnn_train_data_with_visual_embedding_0.3.pt'
# data_path = 'gnn_data/train.pt'
data_path = 'gnn_data/train_sin_arch4_edge_02.pt'
# val_data_path = 'data/gnn_val_data_with_visual_embedding_0.3.pt'
# val_data_path = 'gnn_data/val.pt'
val_data_path = 'gnn_data/val_sin_arch4_edge_02.pt'
N_CLASSES = 49
ablation_suffix = "BasicGATWithEdge"
if not USE_EDGE_FEATURES: ablation_suffix += "_NoEdge"
if not USE_TOPO_LOSS:     ablation_suffix += "_NoTopo"
if not USE_ASYMMETRIC_LOSS: ablation_suffix += "_NoAsym"
if ablation_suffix == "": ablation_suffix = "_FullModel"

# 动态覆盖你原来的 model_save_dir
model_save_dir = f"./checkpoints/AnatomyGAT{ablation_suffix}"
os.makedirs(model_save_dir, exist_ok=True)

CLASS_NAMES = [
    '11', '12', '13', '14', '15', '16', '17',
    '21', '22', '23', '24', '25', '26', '27',
    '31', '32', '33', '34', '35', '36', '37',
    '41', '42', '43', '44', '45', '46', '47',
    '51', '52', '53', '54', '55',
    '61', '62', '63', '64', '65',
    '71', '72', '73', '74', '75',
    '81', '82', '83', '84', '85'
]

# 超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 6  # 可以根据你的显存调整
EPOCHS = 200

PATIENCE_EPOCHS = 20

WEIGHT_DECAY = 1e-5

topo_criterion = TopologicalConsistencyLoss(class_names=CLASS_NAMES, device=device)


def train_epoch(model, loader, criterion, optimizer, current_epoch, lambda_dups=0.5, lambda_fpc=0.5,
                use_topo_loss=True):
    model.train()
    total_loss = 0

    if use_topo_loss:
        if current_epoch <= 15:
            lambda_dups = 0.0
            lambda_fpc = 0.0

        elif current_epoch <= 25:

            lambda_dups, lambda_fpc = 0.1, 0.1
        elif current_epoch <= 35:

            lambda_dups, lambda_fpc = 0.5, 0.5
        else:
            lambda_dups, lambda_fpc = 0.05, 0.05
    else:
        lambda_dups = 0.0
        lambda_fpc = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()


        all_step_logits, attention_weights, model_returned_edge_index, model_returned_edge_type = model(batch)

        batch_loss = 0
        y_true = batch.y.long()

        loss_weights = [0.5, 0.8, 1.0]

        for i, logits in enumerate(all_step_logits):
            weight = loss_weights[i] if i < len(loss_weights) else 0.1

            ce_loss = criterion(logits, y_true)


            l_dups, l_fpc = topo_criterion(
                logits,
                model_returned_edge_index,
                model_returned_edge_type,
                attention_weights
            )

            # 联合损失函数
            step_loss = ce_loss + (lambda_dups * l_dups) + (lambda_fpc * l_fpc)
            batch_loss += weight * step_loss

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, use_topo_loss=True):  # [修复 1] 英文逗号
    model.eval()
    total_loss = 0
    correct_nodes_total = 0
    total_nodes_total = 0
    correct_nodes_confusion = 0
    total_nodes_confusion = 0

    for batch in loader:
        batch = batch.to(device)

        outputs = model(batch)
        topo_loss_val = 0.0  # 初始化

        if isinstance(outputs, tuple):
            all_step_logits = outputs[0]  # 取出 logits 列表
            if len(outputs) > 1:
                topo_tensor = outputs[1]
                if isinstance(topo_tensor, torch.Tensor):
                    topo_loss_val = topo_tensor.mean().item()  # 先 mean() 再 item()
                else:
                    topo_loss_val = float(topo_tensor)
        else:
            all_step_logits = outputs

        out = all_step_logits[-1]  # 取最后一步的 logits

        # 计算基础分类损失
        cls_loss = criterion(out, batch.y.long())


        if use_topo_loss:
            lambda_topo = 0.5
            batch_loss = cls_loss + lambda_topo * topo_loss_val
        else:
            batch_loss = cls_loss

        # 累加批次 Loss
        total_loss += batch_loss.item() * batch.num_graphs

        pred = out.argmax(dim=1)
        y_true = batch.y.long()

        # --- 下面的准确率计算代码保持不变 ---
        correct_nodes_total += int((pred == y_true).sum())
        total_nodes_total += batch.num_nodes

        mask = torch.tensor([y_item.item() in CONFUSION_SET_CLASSES for y_item in y_true.cpu()],
                            dtype=torch.bool).to(device)
        if mask.sum() > 0:
            correct_nodes_confusion += int((pred[mask] == batch.y[mask]).sum())
            total_nodes_confusion += int(mask.sum())

    avg_loss = total_loss / len(loader.dataset)
    accuracy_total = (correct_nodes_total / total_nodes_total) if total_nodes_total > 0 else 0
    accuracy_confusion = (correct_nodes_confusion / total_nodes_confusion) if total_nodes_confusion > 0 else 0

    return avg_loss, accuracy_total, accuracy_confusion


def plot_curves(train_loss_hist, val_loss_hist, val_acc_hist, val_acc_confusion_hist, save_path):
    """绘制所有曲线并将其保存到指定路径。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 绘制训练和验证损失
    ax1.plot(train_loss_hist, label='Train Loss')
    ax1.plot(val_loss_hist, label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制验证准确率
    ax2.plot(val_acc_hist, label='Overall Val Accuracy', color='green')
    ax2.plot(val_acc_confusion_hist, label='Confusion-Set Val Acc.', color='red', linestyle='--')  # 核心指标
    ax2.set_title('Validation Accuracies')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)  # 保存到指定的路径
    logging.info(f"训练曲线图已保存到: {save_path}")
    plt.close(fig)  # 关闭图像，防止在循环中占用过多内存


def main():
    log_file_path = os.path.join(model_save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # 保存到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )

    logging.info(f"使用设备: {device}")
    logging.info(f"实验结果将保存在: {model_save_dir}")
    logging.info(f"监控的混淆集类别: {CONFUSION_SET_CLASSES}")


    model = RecurrentAnatomyGATNew_A(n_classes=N_CLASSES, num_relations=4, num_iterations=1, use_prior=True,
                                     use_edge_features=True, spatial_only=True).to(device)
    logging.info(f"模型 {type(model).__name__} 已实例化 (n_classes={N_CLASSES})")


    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数量: {total_params:,}")

    logging.info(f"训练数据文件为：{data_path}")
    logging.info(f"测试数据文件为：{val_data_path}")


    try:
        train_data_list = torch.load(data_path, map_location='cpu', weights_only=False)
        val_data_list = torch.load(val_data_path, map_location='cpu', weights_only=False)
    except FileNotFoundError as e:
        logging.error(f"数据文件未找到: {e}")
        logging.error("请确保 'gnn_data/train.pt' 和 'gnn_data/val.pt' 存在。")
        return
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        logging.error("请确保您的 .pt 文件包含 torch_geometric.data.Data 对象列表")
        return

    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False)
    logging.info(f"训练数据: {len(train_data_list)} 个图, 验证数据: {len(val_data_list)} 个图")


    alpha_weights = calculate_weights(train_data_list).to(device)
    if USE_ASYMMETRIC_LOSS:
        gamma_pos, gamma_neg = 1.0, 3.0  # 启用非对称 (严惩 FN，忽略容易的背景)
        logging.info("Ablation [ON]: 启用非对称 Focal Loss (gamma_pos=1.0, gamma_neg=3.0)")
    else:
        gamma_pos, gamma_neg = 2.0, 2.0  # 退化为标准对称 Focal Loss
        logging.info("Ablation [OFF]: 禁用非对称，退化为标准 Focal Loss (gamma=2.0)")
    criterion = FocalLoss(alpha=alpha_weights, gamma_pos=gamma_pos, gamma_neg=gamma_neg, bg_idx=48)
    logging.info("使用 FocalLoss (gamma=2), 已计算类别权重")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logging.info(f"使用 AdamW 优化器, LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
        patience=6,
        min_lr=1e-7
    )
    logging.info(f"使用 ReduceLROnPlateau 调度器, 监控模式: max (临床综合得分), Patience={scheduler.patience}")
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_accuracy_confusion_history = []

    logging.info("--- 训练开始 ---")
    best_clinical_score = -float('inf')
    best_val_loss = float('inf')

    early_stop_counter = 0

    last_checkpoints = []
    max_keep = 5

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, use_topo_loss=USE_TOPO_LOSS)
        val_loss, val_accuracy, val_acc_confusion = validate_epoch(model, val_loader, criterion,
                                                                   use_topo_loss=USE_TOPO_LOSS)

        current_lr = optimizer.param_groups[0]['lr']
        clinical_score = (0.5 * val_acc_confusion) + (0.5 * val_accuracy) - (0.1 * val_loss)
        scheduler.step(best_clinical_score)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        val_accuracy_confusion_history.append(val_acc_confusion)

        log_message = (
            f"Epoch: {epoch:03d}, LR: {current_lr:.1e}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, [!!] Confusion Acc: {val_acc_confusion:.4f} [!!]"
        )
        logging.info(log_message)

        is_best_model = False

        if clinical_score > best_clinical_score:
            is_best_model = True

        if is_best_model:
            best_clinical_score = clinical_score
            best_val_confusion_acc = val_acc_confusion
            best_val_acc = val_accuracy
            best_val_loss = val_loss

            early_stop_counter = 0  # 重置早停

            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
            logging.info(f"** 新的最佳模型已保存 **")
            logging.info(
                f" -> 综合得分: {best_clinical_score:.4f} (Conf_Acc: {best_val_confusion_acc:.4f}, Val_Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            logging.info(
                f"综合得分未提升 (当前: {clinical_score:.4f}, 历史最佳: {best_clinical_score:.4f})。早停: {early_stop_counter}/{PATIENCE_EPOCHS}")


        current_ckpt_name = f'model_epoch_{epoch:03d}.pth'
        current_ckpt_path = os.path.join(model_save_dir, current_ckpt_name)

        torch.save(model.state_dict(), current_ckpt_path)
        last_checkpoints.append(current_ckpt_path)

        if len(last_checkpoints) > max_keep:
            oldest_ckpt = last_checkpoints.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)

        # ==========================================
        # 轨道三：早停触发
        # ==========================================
        if early_stop_counter >= PATIENCE_EPOCHS:
            logging.info(f"*** 早停触发：在 {PATIENCE_EPOCHS} 个 epochs 内综合得分未改善。停止训练。 ***")
            break

    logging.info(f"--- 训练结束 ---")
    logging.info(f"最佳综合得分为: {best_clinical_score:.4f} (此时 Val Loss: {best_val_loss:.4f})")

    # 将曲线图保存在与模型相同的目录中
    plot_save_path = os.path.join(model_save_dir, 'training_curves.png')
    plot_curves(
        train_loss_history,
        val_loss_history,
        val_accuracy_history,
        val_accuracy_confusion_history,
        plot_save_path
    )


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    main()