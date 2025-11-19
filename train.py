import logging
import os
import torch.optim
from model import GAT_Numbering_Corrector, GAT_Numbering_Corrector_V2, GAT_Numbering_Corrector_V3,BaselineGNN,AnatomyGAT,RecurrentAnatomyGAT
from focal_loss import FocalLoss
from util import calculate_weights
from torch_geometric.loader import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt


CONFUSION_SET_CLASSES = {
    11, 12, # 上颌右侧切牙
    21, 22, # 上颌左侧切牙
    31, 32, # 下颌左侧切牙
    41, 42, # 下颌右侧切牙
    51, 52,
    61,62,
    71,72,
    81,82,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_path = 'data/gnn_train_data_with_visual_embedding_0.3.pt'
#data_path = 'gnn_data/train.pt'
data_path = 'gnn_data/train_v2.pt'
#val_data_path = 'data/gnn_val_data_with_visual_embedding_0.3.pt'
#val_data_path = 'gnn_data/val.pt'
val_data_path = 'gnn_data/val_v2.pt'
N_CLASSES = 49
model_save_dir = f'checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(model_save_dir, exist_ok=True)

# 超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 16 # 可以根据你的显存调整
EPOCHS = 200
#HIDDEN_CHANNELS = 128
#HEADS = 4
# 1030
#INPUT_CHANNELS = 1033
PATIENCE_EPOCHS = 10
#INPUT_CHANNELS = 55
#OUTPUT_CHANNELS = 49
WEIGHT_DECAY = 1e-5

def train_epoch(model,loader,criterion,optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        all_step_logits = model(batch)

        batch_loss = 0
        y_true = batch.y.long()

        for logits in all_step_logits:
            batch_loss += criterion(logits,y_true)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model,loader,criterion):
    model.eval()
    total_loss = 0

    correct_nodes_total = 0
    total_nodes_total = 0

    # 混淆集准确率指标
    correct_nodes_confusion = 0
    total_nodes_confusion = 0

    for batch in loader:
        batch = batch.to(device)
        all_step_logits = model(batch)

        out = all_step_logits[-1]

        loss = criterion(out, batch.y.long())
        total_loss += loss.item() * batch.num_graphs

        pred = out.argmax(dim=1)

        y_true = batch.y.long()
        # 1. 计算总体准确率
        correct_nodes_total += int((pred == y_true).sum())
        total_nodes_total += batch.num_nodes

        # 2. 计算混淆集准确率 (如我们详细讨论的)
        # 找到所有真实标签属于“混淆集”的节点
        mask = torch.tensor([y_item.item() in CONFUSION_SET_CLASSES for y_item in y_true.cpu()],
                            dtype=torch.bool).to(device)
        if mask.sum() > 0:
            correct_nodes_confusion += int((pred[mask] == batch.y[mask]).sum())
            total_nodes_confusion += int(mask.sum())

    avg_loss = total_loss / len(loader.dataset)

    # 检查分母是否为0
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
    plt.close(fig) # 关闭图像，防止在循环中占用过多内存


def main():
    # [!! 核心 !!] 设置日志记录
    # 日志将同时输出到文件 (training.log) 和控制台 (StreamHandler)
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

    # --- 1. 实例化模型 ---
    #model = BaselineGNN(n_classes=N_CLASSES).to(device)
    #model = AnatomyGAT(n_classes=N_CLASSES,num_relations=3).to(device)
    model = RecurrentAnatomyGAT(n_classes=N_CLASSES,num_relations=3,num_iterations=3).to(device)
    logging.info(f"模型 {type(model).__name__} 已实例化 (n_classes={N_CLASSES})")

    # 打印模型结构 (可选，但有助于调试)
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数量: {total_params:,}")

    # --- 2. 准备数据 ---
    try:
        # map_location=device 可以在加载时就将数据移动到GPU (如果可用)
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

    # --- 3. 准备 Loss 和 优化器 ---
    alpha_weights = calculate_weights(train_data_list).to(device)
    criterion = FocalLoss(alpha_weights, 2)
    logging.info("使用 FocalLoss (gamma=2), 已计算类别权重")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logging.info(f"使用 AdamW 优化器, LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控 val_loss
        factor=0.2,  # 降低力度
        patience=5,  # 稍微缩短 scheduler 的耐心
        min_lr=1e-7
    )

    # 历史记录
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_accuracy_confusion_history = []

    logging.info("--- 训练开始 ---")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, val_acc_confusion = validate_epoch(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # 记录历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        val_accuracy_confusion_history.append(val_acc_confusion)

        # 打印日志
        log_message = (
            f"Epoch: {epoch:03d}, LR: {current_lr:.1e}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, [!!] Confusion Acc: {val_acc_confusion:.4f} [!!]"
        )
        logging.info(log_message)

        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"** 新的最佳模型已保存 (Val Loss: {best_val_loss:.4f}) **")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"Val Loss 未提升。早停计数: {patience_counter}/{PATIENCE_EPOCHS}")

        if patience_counter >= PATIENCE_EPOCHS:
            logging.info(f"*** 早停触发：在 {PATIENCE_EPOCHS} 个 epochs 内验证损失未改善。停止训练。 ***")
            break

    logging.info(f"--- 训练结束 ---")
    logging.info(f"最佳验证损失为: {best_val_loss:.4f}")

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
    # 确保文件夹存在 (如果不存在)
    os.makedirs('checkpoints', exist_ok=True)
    main()