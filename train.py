import logging
import os
import time

import torch.optim

from .model import GAT_Numbering_Corrector
from .focal_loss import FocalLoss
from .util import calculate_weights
from torch_geometric.data import Data, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = 'gnn_train_data.pt'
val_data_path = 'gnn_val_data.pt'
model_save_dir = 'checkpoints'
os.mkdir(model_save_dir,exist_ok = True)


# 超参数
LEARNING_RATE = 1e-3
BATCH_SIZE = 32 # 可以根据你的显存调整
EPOCHS = 500
HIDDEN_CHANNELS = 128
HEADS = 4
INPUT_CHANNELS = 55
OUTPUT_CHANNELS = 49



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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f"使用设备:{device}")


    model = GAT_Numbering_Corrector(
        in_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        heads= HEADS
    ).to(device)
    # build data
    train_data_list = torch.load(data_path)
    val_data_list = torch.load(val_data_path)

    train_loader = DataLoader(train_data_list,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_data_list,batch_size=BATCH_SIZE,shuffle=False)


    alpha_weights = calculate_weights(train_data_list).to(device)
    criterion = FocalLoss(alpha_weights,2)

    optimizer = torch.optim.Adam(model.parameters(),lr= LEARNING_RATE)

    logging.info("训练开始")
    best_val_accuracy = 0.0

    for epoch in range(1,EPOCHS +1):

        train_loss = train_epoch(model,train_loader,criterion,optimizer)
        val_loss,val_accuracy = validate_epoch(model,val_loader,criterion)

        if epoch % 10 == 0:
            logging.info(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 如果当前模型的验证集准确率是历史最高的，就保存它
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"** 新的最佳模型已保存到 {best_model_path} (Accuracy: {best_val_accuracy:.4f}) **")

if __name__ == '__main__':
    main()