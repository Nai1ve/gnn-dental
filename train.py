import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import SlotQualityRecurrentAnatomyGAT


N_CLASSES = 49
BACKGROUND_IDX = 48
NUM_RELATIONS = 4
DEFAULT_TRAIN_DATA = "gnn_data/train_sin_arch4_edge_01_slot_struct.pt"
DEFAULT_VAL_DATA = "gnn_data/val_sin_arch4_edge_01_slot_struct.pt"
DEFAULT_SAVE_DIR = "checkpoints/SlotQuality_StructPrior_NoVisual_Edge_Geom_Prior_T5"

CLASS_NAMES = [
    "11", "12", "13", "14", "15", "16", "17",
    "21", "22", "23", "24", "25", "26", "27",
    "31", "32", "33", "34", "35", "36", "37",
    "41", "42", "43", "44", "45", "46", "47",
    "51", "52", "53", "54", "55",
    "61", "62", "63", "64", "65",
    "71", "72", "73", "74", "75",
    "81", "82", "83", "84", "85",
]
DENTITION_GROUPS = tuple(1 if int(name) // 10 >= 5 else 0 for name in CLASS_NAMES) + (2,)

CONFUSION_SET_CLASSES = {
    11, 12, 21, 22, 31, 32, 41, 42,
    51, 52, 61, 62, 71, 72, 81, 82,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train no-visual slot/quality recurrent anatomy GAT.")
    parser.add_argument("--train-data", default=os.environ.get("GNN_TRAIN_DATA", DEFAULT_TRAIN_DATA))
    parser.add_argument("--val-data", default=os.environ.get("GNN_VAL_DATA", DEFAULT_VAL_DATA))
    parser.add_argument("--save-dir", default=os.environ.get("GNN_SAVE_DIR", DEFAULT_SAVE_DIR))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--geom-dim", type=int, default=37)
    parser.add_argument("--edge-attr-dim", type=int, default=16)
    return parser.parse_args()


def get_raw_labels(batch):
    if hasattr(batch, "pred_labels_raw"):
        return batch.pred_labels_raw.long().clamp(0, BACKGROUND_IDX)
    return batch.x_prior[:, :N_CLASSES].argmax(dim=1).long().clamp(0, BACKGROUND_IDX)


def get_slot_targets(batch):
    if hasattr(batch, "slot_y"):
        return batch.slot_y.long()
    return torch.where(batch.y.long() == BACKGROUND_IDX, torch.full_like(batch.y.long(), -100), batch.y.long())


def get_quality_targets(batch):
    if hasattr(batch, "quality_y"):
        return batch.quality_y.long()
    return (batch.y.long() != BACKGROUND_IDX).long()


def get_detector_scores(batch):
    if hasattr(batch, "pred_scores_raw"):
        return batch.pred_scores_raw.float().clamp(0, 1)
    return ((batch.x_prior[:, -1].float() + 1.0) / 2.0).clamp(0, 1)


def get_dentition_groups(labels):
    lookup = torch.as_tensor(DENTITION_GROUPS, dtype=torch.long, device=labels.device)
    return lookup[labels.clamp(0, BACKGROUND_IDX)]


def step_to_logits(step_output):
    slot_logits = step_output["slot_logits"]
    quality_logits = step_output["quality_logits"]
    keep_logit = quality_logits[:, 1:2]
    drop_logit = quality_logits[:, 0:1]
    class_logits = torch.cat([slot_logits + keep_logit, drop_logit], dim=1)
    return slot_logits, quality_logits, class_logits


def weighted_mean(values, weights):
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def margin_loss(logits, target, competitor, mask, margin):
    if mask.sum() == 0:
        return logits.sum() * 0.0
    target_logit = logits[mask, target[mask]]
    competitor_logit = logits[mask, competitor[mask]]
    return F.relu(margin - target_logit + competitor_logit).mean()


class SlotQualityCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, step_output, batch):
        slot_logits, quality_logits, class_logits = step_to_logits(step_output)
        slot_target = get_slot_targets(batch).to(slot_logits.device)
        quality_target = get_quality_targets(batch).to(slot_logits.device)
        raw_labels = get_raw_labels(batch).to(slot_logits.device)
        detector_scores = get_detector_scores(batch).to(slot_logits.device)

        slot_valid = (slot_target >= 0) & (slot_target < BACKGROUND_IDX)
        safe_slot_target = slot_target.clamp(0, BACKGROUND_IDX - 1)

        slot_ce_all = F.cross_entropy(slot_logits, safe_slot_target, reduction="none")
        raw_correct = slot_valid & (raw_labels == safe_slot_target)
        raw_wrong = slot_valid & (raw_labels != safe_slot_target)
        high_score_correct = raw_correct & (detector_scores >= 0.80)
        slot_weights = torch.zeros_like(slot_ce_all)
        slot_weights[raw_correct] = 1.2
        slot_weights[high_score_correct] = 1.8
        slot_weights[raw_wrong] = 2.8
        slot_loss = weighted_mean(slot_ce_all, slot_weights) if slot_valid.any() else slot_ce_all.sum() * 0.0

        quality_ce = F.cross_entropy(quality_logits, quality_target.clamp(0, 1), reduction="none")
        quality_weights = torch.full_like(quality_ce, 0.35)
        quality_weights[quality_target == 1] = 1.2
        if hasattr(batch, "node_role"):
            node_role = batch.node_role.to(slot_logits.device)
            quality_weights[node_role == 1] = 1.0
            quality_weights[node_role == 2] = 0.8
            quality_weights[node_role == 3] = 0.7
        quality_loss = weighted_mean(quality_ce, quality_weights)

        keep_mask = raw_correct & (quality_target == 1)
        keep_competitor = torch.full_like(safe_slot_target, BACKGROUND_IDX)
        keep_margin = margin_loss(class_logits, safe_slot_target, keep_competitor, keep_mask, margin=0.35)

        correction_mask = raw_wrong & (raw_labels < BACKGROUND_IDX)
        correction_margin = margin_loss(slot_logits, safe_slot_target, raw_labels.clamp(0, BACKGROUND_IDX - 1),
                                        correction_mask, margin=0.20)

        raw_dentition = get_dentition_groups(raw_labels)
        target_dentition = get_dentition_groups(safe_slot_target)
        cross_dentition_wrong = raw_correct & (raw_labels < BACKGROUND_IDX)
        cross_dentition_penalty = margin_loss(
            slot_logits,
            raw_labels.clamp(0, BACKGROUND_IDX - 1),
            slot_logits[:, :BACKGROUND_IDX].argmax(dim=1),
            cross_dentition_wrong
            & (raw_dentition != get_dentition_groups(slot_logits[:, :BACKGROUND_IDX].argmax(dim=1)))
            & (detector_scores >= 0.60),
            margin=0.30,
        )
        cross_dentition_correction = correction_mask & (raw_dentition != target_dentition)
        cross_dentition_correction_margin = margin_loss(
            slot_logits,
            safe_slot_target,
            raw_labels.clamp(0, BACKGROUND_IDX - 1),
            cross_dentition_correction,
            margin=0.35,
        )

        keep_vs_drop = F.relu(0.25 - quality_logits[:, 1] + quality_logits[:, 0])
        drop_vs_keep = F.relu(0.10 - quality_logits[:, 0] + quality_logits[:, 1])
        quality_margin = (
            weighted_mean(keep_vs_drop, (quality_target == 1).float())
            + 0.5 * weighted_mean(drop_vs_keep, (quality_target == 0).float())
        )

        total = (
            slot_loss
            + 0.55 * quality_loss
            + 0.10 * keep_margin
            + 0.15 * correction_margin
            + 0.08 * cross_dentition_penalty
            + 0.05 * cross_dentition_correction_margin
            + 0.05 * quality_margin
        )
        parts = {
            "slot": slot_loss.detach(),
            "quality": quality_loss.detach(),
            "keep_margin": keep_margin.detach(),
            "correction_margin": correction_margin.detach(),
            "cross_dentition": cross_dentition_penalty.detach(),
            "cross_dentition_correction": cross_dentition_correction_margin.detach(),
            "quality_margin": quality_margin.detach(),
        }
        return total, parts


def conservative_relabel_for_validation(class_logits, raw_labels, margin=0.06, min_prob=0.20):
    probs = F.softmax(class_logits, dim=1)
    raw_labels = raw_labels.clamp(0, BACKGROUND_IDX)
    best_tooth_probs, best_tooth_labels = probs[:, :BACKGROUND_IDX].max(dim=1)
    raw_probs = probs.gather(1, raw_labels.unsqueeze(1)).squeeze(1)
    relabel_mask = (
        (best_tooth_labels != raw_labels)
        & (best_tooth_probs >= min_prob)
        & ((best_tooth_probs - raw_probs) >= margin)
    )
    pred = raw_labels.clone()
    pred[relabel_mask] = best_tooth_labels[relabel_mask]
    return pred


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    part_keys = [
        "slot",
        "quality",
        "keep_margin",
        "correction_margin",
        "cross_dentition",
        "cross_dentition_correction",
        "quality_margin",
    ]
    totals = {"loss": 0.0, **{key: 0.0 for key in part_keys}}
    loss_weights = torch.tensor([0.2, 0.35, 0.5, 0.75, 1.0], device=device)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        all_step_outputs = model(batch)
        if len(all_step_outputs) != len(loss_weights):
            weights = torch.linspace(0.2, 1.0, steps=len(all_step_outputs), device=device)
        else:
            weights = loss_weights

        batch_loss = 0.0
        last_parts = None
        for weight, step_output in zip(weights, all_step_outputs):
            step_loss, parts = criterion(step_output, batch)
            batch_loss = batch_loss + weight * step_loss
            last_parts = parts

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scale = batch.num_graphs
        totals["loss"] += batch_loss.item() * scale
        for key in part_keys:
            totals[key] += last_parts[key].item() * scale

    dataset_size = len(loader.dataset)
    return {key: value / dataset_size for key, value in totals.items()}


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_total = 0
    total_nodes = 0
    correct_confusion = 0
    total_confusion = 0
    raw_wrong_tooth = 0
    corrected_tooth = 0
    raw_correct_tooth = 0
    bad_change = 0
    cross_dentition_bad_change = 0
    fp_h_keep = 0
    fp_h_total = 0
    pred_bg_for_tooth = 0
    tooth_total = 0

    for batch in loader:
        batch = batch.to(device)
        all_step_outputs = model(batch)
        step_output = all_step_outputs[-1]
        loss, _ = criterion(step_output, batch)
        total_loss += loss.item() * batch.num_graphs

        _, _, class_logits = step_to_logits(step_output)
        _, quality_logits, _ = step_to_logits(step_output)
        raw_labels = get_raw_labels(batch).to(device)
        pred = conservative_relabel_for_validation(class_logits, raw_labels)
        y_true = batch.y.long().to(device)
        quality_target = get_quality_targets(batch).to(device)
        tooth_mask = y_true != BACKGROUND_IDX

        correct_total += int((pred == y_true).sum())
        total_nodes += batch.num_nodes

        raw_correct = tooth_mask & (raw_labels == y_true)
        raw_wrong = tooth_mask & (raw_labels != y_true)
        raw_correct_tooth += int(raw_correct.sum())
        raw_wrong_tooth += int(raw_wrong.sum())
        corrected_tooth += int((raw_wrong & (pred == y_true)).sum())
        bad_change += int((raw_correct & (pred != y_true)).sum())
        cross_bad = (
            raw_correct
            & (pred != y_true)
            & (get_dentition_groups(raw_labels) != get_dentition_groups(pred))
        )
        cross_dentition_bad_change += int(cross_bad.sum())
        pred_bg_for_tooth += int((tooth_mask & (pred == BACKGROUND_IDX)).sum())
        tooth_total += int(tooth_mask.sum())

        if hasattr(batch, "node_role"):
            node_role = batch.node_role.to(device)
            fp_h_mask = (quality_target == 0) & (node_role == 3)
            if fp_h_mask.any():
                fp_h_keep += int((quality_logits[fp_h_mask, 1] >= quality_logits[fp_h_mask, 0]).sum())
                fp_h_total += int(fp_h_mask.sum())

        confusion_mask = torch.zeros_like(y_true, dtype=torch.bool)
        for cls_idx in CONFUSION_SET_CLASSES:
            confusion_mask |= y_true == cls_idx
        if confusion_mask.any():
            correct_confusion += int((pred[confusion_mask] == y_true[confusion_mask]).sum())
            total_confusion += int(confusion_mask.sum())

    diagnostics = {
        "correction_rate": corrected_tooth / raw_wrong_tooth if raw_wrong_tooth else 0.0,
        "bad_change_rate": bad_change / raw_correct_tooth if raw_correct_tooth else 0.0,
        "pred_bg_for_tooth_rate": pred_bg_for_tooth / tooth_total if tooth_total else 0.0,
        "cross_dentition_bad_change_rate": (
            cross_dentition_bad_change / raw_correct_tooth if raw_correct_tooth else 0.0
        ),
        "fp_h_keep_rate": fp_h_keep / fp_h_total if fp_h_total else 0.0,
        "raw_wrong_tooth_total": raw_wrong_tooth,
        "corrected_tooth_total": corrected_tooth,
        "bad_change_total": bad_change,
        "cross_dentition_bad_change_total": cross_dentition_bad_change,
        "fp_h_keep_total": fp_h_keep,
        "fp_h_total": fp_h_total,
    }
    return (
        total_loss / len(loader.dataset),
        correct_total / total_nodes if total_nodes else 0.0,
        correct_confusion / total_confusion if total_confusion else 0.0,
        diagnostics,
    )


def plot_curves(train_loss_hist, val_loss_hist, val_acc_hist, val_acc_confusion_hist, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.plot(train_loss_hist, label="Train Loss")
    ax1.plot(val_loss_hist, label="Validation Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(val_acc_hist, label="Overall Val Accuracy")
    ax2.plot(val_acc_confusion_hist, label="Confusion-Set Val Acc.", linestyle="--")
    ax2.set_title("Validation Accuracies")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    log_file_path = os.path.join(args.save_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    logging.info(f"device={device}")
    logging.info(f"train_data={args.train_data}")
    logging.info(f"val_data={args.val_data}")
    logging.info(f"save_dir={args.save_dir}")
    logging.info(
        f"training_config: T={args.num_iterations}, use_visual=False, "
        f"use_geom=True, use_prior=True, use_edge_features=True, spatial_only=False"
    )

    train_data = torch.load(args.train_data, map_location="cpu", weights_only=False)
    val_data = torch.load(args.val_data, map_location="cpu", weights_only=False)
    logging.info(f"train graphs={len(train_data)}, val graphs={len(val_data)}")

    if train_data:
        args.geom_dim = int(train_data[0].x_geom.size(1))
        args.edge_attr_dim = int(train_data[0].edge_attr_overlap.size(1))
        logging.info(f"inferred geom_dim={args.geom_dim}, edge_attr_dim={args.edge_attr_dim}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = SlotQualityRecurrentAnatomyGAT(
        n_classes=N_CLASSES,
        num_relations=NUM_RELATIONS,
        num_iterations=args.num_iterations,
        use_visual=False,
        use_geom=True,
        use_prior=True,
        use_edge_features=True,
        spatial_only=False,
        geom_dim=args.geom_dim,
        edge_attr_dim=args.edge_attr_dim,
    ).to(device)
    criterion = SlotQualityCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=6, min_lr=1e-7
    )

    best_score = -float("inf")
    best_summary = {}
    early_stop = 0
    train_loss_hist, val_loss_hist, val_acc_hist, val_conf_hist = [], [], [], []
    last_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_conf_acc, val_diag = validate_epoch(model, val_loader, criterion, device)

        score = (
            0.40 * val_acc
            + 0.35 * val_conf_acc
            + 1.25 * val_diag["correction_rate"]
            - 1.25 * val_diag["bad_change_rate"]
            - 0.75 * val_diag["cross_dentition_bad_change_rate"]
            - 0.30 * val_diag["fp_h_keep_rate"]
            - 0.35 * val_diag["pred_bg_for_tooth_rate"]
            - 0.05 * val_loss
        )
        scheduler.step(score)

        train_loss_hist.append(train_metrics["loss"])
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_conf_hist.append(val_conf_acc)

        logging.info(
            f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.1e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(slot={train_metrics['slot']:.4f}, quality={train_metrics['quality']:.4f}, "
            f"cross={train_metrics['cross_dentition']:.4f}) | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, conf_acc={val_conf_acc:.4f}, "
            f"correction_rate={val_diag['correction_rate']:.4f}, "
            f"bad_change_rate={val_diag['bad_change_rate']:.4f}, "
            f"cross_bad_rate={val_diag['cross_dentition_bad_change_rate']:.4f}, "
            f"fp_h_keep_rate={val_diag['fp_h_keep_rate']:.4f}, "
            f"pred_bg_tooth_rate={val_diag['pred_bg_for_tooth_rate']:.4f}, score={score:.4f}"
        )

        bad_change_ok = val_diag["bad_change_rate"] <= 0.015
        cross_bad_ok = val_diag["cross_dentition_bad_change_rate"] <= 0.010
        if bad_change_ok and cross_bad_ok and score > best_score:
            best_score = score
            best_summary = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_conf_acc": val_conf_acc,
                **val_diag,
            }
            early_stop = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            logging.info(f"** best model saved: score={best_score:.4f}, summary={best_summary}")
        else:
            early_stop += 1
            logging.info(
                f"no best update, early_stop={early_stop}/{args.patience}, "
                f"bad_change_ok={bad_change_ok}, cross_bad_ok={cross_bad_ok}"
            )

        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        last_checkpoints.append(ckpt_path)
        if len(last_checkpoints) > 5:
            old = last_checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)

        if early_stop >= args.patience:
            logging.info("early stopping triggered")
            break

    logging.info(f"training finished. best_score={best_score:.4f}, best_summary={best_summary}")
    plot_curves(
        train_loss_hist,
        val_loss_hist,
        val_acc_hist,
        val_conf_hist,
        os.path.join(args.save_dir, "training_curves.png"),
    )


if __name__ == "__main__":
    main()
