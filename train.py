import logging
import os
import torch.optim
from model import GAT_Numbering_Corrector, GAT_Numbering_Corrector_V2, GAT_Numbering_Corrector_V3
from focal_loss import FocalLoss
from util import calculate_weights
from torch_geometric.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_path = 'data/gnn_train_data_with_visual_embedding_0.3.pt'
data_path = 'gnn_data/gnn_train_data_CLEANED.pt'
#val_data_path = 'data/gnn_val_data_with_visual_embedding_0.3.pt'
val_data_path = 'gnn_data/gnn_val_data_with_visual_embedding_score0.05_iou0.50_k9.pt'

model_save_dir = f'checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_V3_16_embedding'
os.makedirs(model_save_dir, exist_ok=True)


# 超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 16 # 可以根据你的显存调整
EPOCHS = 100
HIDDEN_CHANNELS = 128
HEADS = 4
# 1030
INPUT_CHANNELS = 1033
PATIENCE_EPOCHS = 10
#INPUT_CHANNELS = 55
OUTPUT_CHANNELS = 49
WEIGHT_DECAY = 1e-5

def train_epoch(model,loader,criterion,optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out,batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model,loader,criterion):
    model.eval()
    total_loss = 0
    correct_nodes = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out,batch.y)
        total_loss += loss.item() * batch.num_graphs

        # Numbering Accuracy
        pred = out.argmax(dim = 1)
        correct_nodes += int((pred == batch.y).sum())
        total_nodes += batch.num_nodes

    avg_loss = total_loss / len(loader.dataset)
    numbering_accuracy = correct_nodes / total_nodes
    return avg_loss,numbering_accuracy

# ADDED: A function to plot the training curves
def plot_curves(train_loss_history, val_loss_history, val_accuracy_history):
    """Plots the training and validation loss, and validation accuracy curves."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plotting training and validation loss
    ax1.plot(train_loss_history, label='Train Loss')
    ax1.plot(val_loss_history, label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting validation accuracy
    ax2.plot(val_accuracy_history, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'training_curves_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    logging.info("训练曲线图已保存到 training_curves_1.png")
    # plt.show() # Uncomment this line if you want to display the plot immediately

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f"使用设备:{device}")

    # model = GAT_Numbering_Corrector(
    #     in_channels=INPUT_CHANNELS,
    #     hidden_channels=HIDDEN_CHANNELS,
    #     out_channels=OUTPUT_CHANNELS,
    #     heads= HEADS
    # ).to(device)

    # model = GAT_Numbering_Corrector_V2(
    #     in_channels=INPUT_CHANNELS,
    #     hidden_channels=HIDDEN_CHANNELS,
    #     out_channels=OUTPUT_CHANNELS,
    #     heads=HEADS,
    #     dropout_rate=0.2,
    # ).to(device)
    #
    model = GAT_Numbering_Corrector_V3(
        in_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        heads= HEADS,
        dropout_rate=0.5
    ).to(device)

    # build data
    train_data_list = torch.load(data_path,weights_only=False)
    val_data_list = torch.load(val_data_path,weights_only=False)

    train_loader = DataLoader(train_data_list,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_data_list,batch_size=BATCH_SIZE,shuffle=False)

    alpha_weights = calculate_weights(train_data_list).to(device)
    criterion = FocalLoss(alpha_weights,2)

    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 当监控的指标停止最大化时，降低学习率
        factor=0.1,  # 学习率降低的倍数 (new_lr = lr * factor)
        patience=10  # 在10个epoch内验证准确率没有提升，就降低学习率

    )

    # ADDED: Lists to store history for plotting
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    logging.info("训练开始")
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1,EPOCHS +1):

        train_loss = train_epoch(model,train_loader,criterion,optimizer)
        val_loss,val_accuracy = validate_epoch(model,val_loader,criterion)

        scheduler.step(val_loss)

        # ADDED: Append the latest metrics to our history lists
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        if epoch % 10 == 0:
            logging.info(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # [!!!] 核心修正：实施早停和模型保存 [!!!]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"** 新的最佳模型已保存 (Val Loss: {best_val_loss:.4f}, Val Acc: {val_accuracy:.4f}) **")
            patience_counter = 0 # 重置计数器
        else:
            patience_counter += 1
            # (可以取消下面这行的注释来获取更详细的日志)
            logging.info(f"Val Loss 未提升。早停计数: {patience_counter}/{PATIENCE_EPOCHS}")

        if patience_counter >= PATIENCE_EPOCHS:
            logging.info(f"*** 早停触发：验证损失在 {PATIENCE_EPOCHS} 个 epochs 内未改善。停止训练。 ***")
            break # 跳出训练循环


        # # 如果当前模型的验证集准确率是历史最高的，就保存它
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     best_model_path = os.path.join(model_save_dir, 'best_model.pth')
        #     torch.save(model.state_dict(), best_model_path)
        #     logging.info(f"** 新的最佳模型已保存到 {best_model_path} (Accuracy: {best_val_accuracy:.4f}) **")

    # ADDED: Call the plotting function after the training loop is finished
    logging.info(f"训练结束。最佳验证损失为: {best_val_loss:.4f}")
    logging.info("训练结束，正在生成曲线图...")
    plot_curves(train_loss_history, val_loss_history, val_accuracy_history)

if __name__ == '__main__':
    main()