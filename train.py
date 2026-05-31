import logging
import os
import torch
import torch.optim
import torch.nn.functional as F
from model import RecurrentAnatomyGATNew_A
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

USE_EDGE_FEATURES = True  # 消融 1: 是否启用动态边缘特征
USE_VISUAL = False
USE_GEOM = True
USE_PRIOR = True
SPATIAL_ONLY = False
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
data_path = 'gnn_data/train_sin_arch4_edge_01.pt'
# val_data_path = 'data/gnn_val_data_with_visual_embedding_0.3.pt'
# val_data_path = 'gnn_data/val.pt'
val_data_path = 'gnn_data/val_sin_arch4_edge_01.pt'
N_CLASSES = 49
BACKGROUND_IDX = 48
ablation_suffix = "CorrectionCE"
if not USE_VISUAL: ablation_suffix += "_NoVisual"
if USE_EDGE_FEATURES: ablation_suffix += "_Edge"
if USE_GEOM: ablation_suffix += "_Geom"
if USE_PRIOR: ablation_suffix += "_Prior"
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
BG_LOW_SCORE_THRESHOLD = 0.20
BG_LABEL_SMOOTHING = 0.05
KEEP_MARGIN = 0.30
BG_MARGIN = 0.15
LAMBDA_KEEP_MARGIN = 0.05
LAMBDA_BG_MARGIN = 0.02
BAD_CHANGE_RATE_LIMIT = 0.03
VAL_RELABEL_MARGIN = 0.15
VAL_RELABEL_MIN_PROB = 0.35


class CorrectionCriterion(torch.nn.Module):
    """
    让 GNN 学“保留正确编号 + 修正错误编号”，background 只作为弱监督。
    """

    def __init__(self, n_classes=N_CLASSES, background_idx=BACKGROUND_IDX):
        super().__init__()
        self.n_classes = n_classes
        self.background_idx = background_idx

    def _prior_labels(self, batch):
        if hasattr(batch, 'pred_labels_raw'):
            return batch.pred_labels_raw.long()
        return batch.x_prior[:, :self.n_classes].argmax(dim=1).long()

    @staticmethod
    def _scores(batch):
        if hasattr(batch, 'pred_scores_raw'):
            return batch.pred_scores_raw.float()
        return ((batch.x_prior[:, -1].float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def _sample_weights(self, batch, y_true):
        raw_label = self._prior_labels(batch).to(y_true.device)
        scores = self._scores(batch).to(y_true.device)

        weights = torch.ones_like(scores, dtype=torch.float)
        is_bg = y_true == self.background_idx
        is_tooth = ~is_bg
        is_correction = (raw_label != y_true) & is_tooth

        weights[is_tooth] = 1.0
        weights[is_correction] = 1.5
        weights[is_bg & (scores < BG_LOW_SCORE_THRESHOLD)] = 0.4
        weights[is_bg & (scores >= BG_LOW_SCORE_THRESHOLD)] = 0.2
        return weights, raw_label, scores

    def _soft_ce(self, logits, y_true, weights):
        log_probs = F.log_softmax(logits, dim=1)
        target = F.one_hot(y_true, num_classes=self.n_classes).float()

        bg_mask = y_true == self.background_idx
        if bg_mask.any() and BG_LABEL_SMOOTHING > 0:
            smooth = BG_LABEL_SMOOTHING / (self.n_classes - 1)
            target[bg_mask] = smooth
            target[bg_mask, self.background_idx] = 1.0 - BG_LABEL_SMOOTHING

        per_node_loss = -(target * log_probs).sum(dim=1)
        return (per_node_loss * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _target_margin_loss(logits, target_labels, mask, margin):
        if not mask.any():
            return logits.new_tensor(0.0)

        selected_logits = logits[mask]
        selected_targets = target_labels[mask]
        target_logits = selected_logits.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
        competitor_logits = selected_logits.clone()
        competitor_logits.scatter_(1, selected_targets.unsqueeze(1), -1e4)
        max_competitor = competitor_logits.max(dim=1).values
        return F.relu(margin + max_competitor - target_logits).mean()

    def forward(self, logits, batch):
        y_true = batch.y.long()
        weights, raw_label, scores = self._sample_weights(batch, y_true)

        ce_loss = self._soft_ce(logits, y_true, weights)

        keep_mask = (raw_label == y_true) & (y_true != self.background_idx)
        keep_margin_loss = self._target_margin_loss(logits, y_true, keep_mask, KEEP_MARGIN)

        high_conf_bg_mask = (y_true == self.background_idx) & (scores >= BG_LOW_SCORE_THRESHOLD)
        bg_margin_loss = self._target_margin_loss(
            logits, torch.full_like(y_true, self.background_idx), high_conf_bg_mask, BG_MARGIN
        )

        total_loss = ce_loss + LAMBDA_KEEP_MARGIN * keep_margin_loss + LAMBDA_BG_MARGIN * bg_margin_loss
        loss_parts = {
            'ce': ce_loss.detach(),
            'keep_margin': keep_margin_loss.detach(),
            'bg_margin': bg_margin_loss.detach()
        }
        return total_loss, loss_parts


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_ce = 0
    total_keep_margin = 0
    total_bg_margin = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()


        all_step_logits, attention_weights, model_returned_edge_index, model_returned_edge_type = model(batch)

        batch_loss = 0
        loss_weights = torch.linspace(0.5, 1.0, steps=len(all_step_logits), device=device)

        for i, logits in enumerate(all_step_logits):
            weight = loss_weights[i]
            step_loss, loss_parts = criterion(logits, batch)
            batch_loss += weight * step_loss

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item() * batch.num_graphs
        total_ce += loss_parts['ce'].item() * batch.num_graphs
        total_keep_margin += loss_parts['keep_margin'].item() * batch.num_graphs
        total_bg_margin += loss_parts['bg_margin'].item() * batch.num_graphs

    dataset_size = len(loader.dataset)
    return (
        total_loss / dataset_size,
        total_ce / dataset_size,
        total_keep_margin / dataset_size,
        total_bg_margin / dataset_size
    )


@torch.no_grad()
def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct_nodes_total = 0
    total_nodes_total = 0
    correct_nodes_confusion = 0
    total_nodes_confusion = 0
    raw_correct_tooth_total = 0
    raw_wrong_tooth_total = 0
    corrected_tooth_total = 0
    bad_change_total = 0
    tooth_total = 0
    pred_bg_for_tooth_total = 0

    for batch in loader:
        batch = batch.to(device)

        outputs = model(batch)

        if isinstance(outputs, tuple):
            all_step_logits = outputs[0]  # 取出 logits 列表
        else:
            all_step_logits = outputs

        out = all_step_logits[-1]  # 取最后一步的 logits

        batch_loss, _ = criterion(out, batch)
        total_loss += batch_loss.item() * batch.num_graphs

        y_true = batch.y.long()
        raw_label = criterion._prior_labels(batch).to(device)
        pred = conservative_relabel_for_validation(out, raw_label)

        correct_nodes_total += int((pred == y_true).sum())
        total_nodes_total += batch.num_nodes

        tooth_mask = y_true != BACKGROUND_IDX
        raw_correct_tooth = tooth_mask & (raw_label == y_true)
        raw_wrong_tooth = tooth_mask & (raw_label != y_true)
        bad_change = raw_correct_tooth & (pred != y_true)
        corrected_tooth = raw_wrong_tooth & (pred == y_true)

        tooth_total += int(tooth_mask.sum())
        raw_correct_tooth_total += int(raw_correct_tooth.sum())
        raw_wrong_tooth_total += int(raw_wrong_tooth.sum())
        corrected_tooth_total += int(corrected_tooth.sum())
        bad_change_total += int(bad_change.sum())
        pred_bg_for_tooth_total += int((tooth_mask & (pred == BACKGROUND_IDX)).sum())

        mask = torch.tensor([y_item.item() in CONFUSION_SET_CLASSES for y_item in y_true.cpu()],
                            dtype=torch.bool).to(device)
        if mask.sum() > 0:
            correct_nodes_confusion += int((pred[mask] == batch.y[mask]).sum())
            total_nodes_confusion += int(mask.sum())

    avg_loss = total_loss / len(loader.dataset)
    accuracy_total = (correct_nodes_total / total_nodes_total) if total_nodes_total > 0 else 0
    accuracy_confusion = (correct_nodes_confusion / total_nodes_confusion) if total_nodes_confusion > 0 else 0
    correction_rate = (corrected_tooth_total / raw_wrong_tooth_total) if raw_wrong_tooth_total > 0 else 0
    bad_change_rate = (bad_change_total / raw_correct_tooth_total) if raw_correct_tooth_total > 0 else 0
    pred_bg_for_tooth_rate = (pred_bg_for_tooth_total / tooth_total) if tooth_total > 0 else 0

    diagnostics = {
        'correction_rate': correction_rate,
        'bad_change_rate': bad_change_rate,
        'pred_bg_for_tooth_rate': pred_bg_for_tooth_rate,
        'raw_wrong_tooth_total': raw_wrong_tooth_total,
        'corrected_tooth_total': corrected_tooth_total,
        'bad_change_total': bad_change_total,
    }
    return avg_loss, accuracy_total, accuracy_confusion, diagnostics


def conservative_relabel_for_validation(logits, raw_labels):
    probs = F.softmax(logits, dim=1)
    raw_labels = raw_labels.to(logits.device).clamp(0, N_CLASSES - 1)
    best_tooth_probs, best_tooth_labels = probs[:, :BACKGROUND_IDX].max(dim=1)
    raw_probs = probs.gather(1, raw_labels.unsqueeze(1)).squeeze(1)

    relabel_mask = (
        (best_tooth_labels != raw_labels)
        & (best_tooth_probs >= VAL_RELABEL_MIN_PROB)
        & ((best_tooth_probs - raw_probs) >= VAL_RELABEL_MARGIN)
    )

    pred = raw_labels.clone()
    pred[relabel_mask] = best_tooth_labels[relabel_mask]
    return pred


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


    model = RecurrentAnatomyGATNew_A(n_classes=N_CLASSES, num_relations=4, num_iterations=1,
                                     use_visual=USE_VISUAL, use_geom=USE_GEOM, use_prior=USE_PRIOR,
                                     use_edge_features=USE_EDGE_FEATURES, spatial_only=SPATIAL_ONLY).to(device)
    logging.info(f"模型 {type(model).__name__} 已实例化 (n_classes={N_CLASSES})")
    logging.info(
        f"训练设置: use_visual={USE_VISUAL}, use_geom={USE_GEOM}, use_prior={USE_PRIOR}, "
        f"use_edge_features={USE_EDGE_FEATURES}, spatial_only={SPATIAL_ONLY}"
    )


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


    criterion = CorrectionCriterion(n_classes=N_CLASSES, background_idx=BACKGROUND_IDX)
    logging.info(
        "使用 CorrectionCriterion: weighted CE + weak background smoothing + light keep/bg margins"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logging.info(f"使用 AdamW 优化器, LR={LEARNING_RATE}, WeightDecay={WEIGHT_DECAY}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
        patience=6,
        min_lr=1e-7
    )
    logging.info(f"使用 ReduceLROnPlateau 调度器, 监控模式: max (val correction score), Patience={scheduler.patience}")
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_accuracy_confusion_history = []

    logging.info("--- 训练开始 ---")
    best_clinical_score = -float('inf')
    best_val_loss = float('inf')
    best_val_confusion_acc = 0.0
    best_val_acc = 0.0
    best_diagnostics = {}

    early_stop_counter = 0

    last_checkpoints = []
    max_keep = 5

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_ce, train_keep_margin, train_bg_margin = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, val_acc_confusion, val_diag = validate_epoch(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]['lr']
        clinical_score = (
            0.45 * val_accuracy
            + 0.45 * val_acc_confusion
            + 0.80 * val_diag['correction_rate']
            - 2.00 * val_diag['bad_change_rate']
            - 0.50 * val_diag['pred_bg_for_tooth_rate']
            - 0.05 * val_loss
        )
        scheduler.step(clinical_score)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        val_accuracy_confusion_history.append(val_acc_confusion)

        log_message = (
            f"Epoch: {epoch:03d}, LR: {current_lr:.1e}, "
            f"Train Loss: {train_loss:.4f} (CE={train_ce:.4f}, KeepM={train_keep_margin:.4f}, BgM={train_bg_margin:.4f}), "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Confusion Acc: {val_acc_confusion:.4f}, "
            f"Correction Rate: {val_diag['correction_rate']:.4f}, Bad Change Rate: {val_diag['bad_change_rate']:.4f}, "
            f"Pred-BG Tooth Rate: {val_diag['pred_bg_for_tooth_rate']:.4f}, Score: {clinical_score:.4f}"
        )
        logging.info(log_message)

        bad_change_ok = val_diag['bad_change_rate'] <= BAD_CHANGE_RATE_LIMIT
        is_best_model = bad_change_ok and (clinical_score > best_clinical_score)
        if best_clinical_score == -float('inf') and clinical_score > best_clinical_score:
            is_best_model = True

        if is_best_model:
            best_clinical_score = clinical_score
            best_val_confusion_acc = val_acc_confusion
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            best_diagnostics = val_diag

            early_stop_counter = 0  # 重置早停

            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
            logging.info(f"** 新的最佳模型已保存 **")
            logging.info(
                f" -> 综合得分: {best_clinical_score:.4f} "
                f"(Conf_Acc: {best_val_confusion_acc:.4f}, Val_Acc: {best_val_acc:.4f}, "
                f"Loss: {best_val_loss:.4f}, Correction_Rate: {best_diagnostics['correction_rate']:.4f}, "
                f"Bad_Change_Rate: {best_diagnostics['bad_change_rate']:.4f})")
        else:
            early_stop_counter += 1
            logging.info(
                f"综合得分未提升或 bad_change_rate 超过约束 "
                f"(当前: {clinical_score:.4f}, 历史最佳: {best_clinical_score:.4f}, "
                f"bad_change_ok={bad_change_ok})。早停: {early_stop_counter}/{PATIENCE_EPOCHS}")


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
    logging.info(
        f"最佳综合得分为: {best_clinical_score:.4f} "
        f"(Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}, "
        f"Conf Acc: {best_val_confusion_acc:.4f}, Diagnostics: {best_diagnostics})"
    )

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
