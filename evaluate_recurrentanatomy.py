import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from model import SlotQualityRecurrentAnatomyGAT


N_CLASSES = 49
BACKGROUND_IDX = 48
NUM_RELATIONS = 5
DEFAULT_MODEL_PATH = "checkpoints/SlotQuality_StructOrder_NoVisual_T5_APAware/best_model.pth"
DEFAULT_TEST_DATA_PATH = "gnn_data/test_ccsw_liu_exp_open_slot_struct_order.pt"
DEFAULT_OUTPUT_NAME = "graph_by_graph_predictions_recurrent_ccsw_slot_quality_struct_order_t5_exp_open.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
JAW_GROUPS = tuple(0 if int(name) // 10 in (1, 2, 5, 6) else 1 for name in CLASS_NAMES) + (2,)
SIDE_GROUPS = tuple(0 if int(name) // 10 in (1, 4, 5, 8) else 1 for name in CLASS_NAMES) + (2,)
TOOTH_INDEX = tuple(int(name) % 10 for name in CLASS_NAMES) + (0,)

CONFUSION_SET_CLASSES = {
    11, 12, 21, 22, 31, 32, 41, 42,
    51, 52, 61, 62, 71, 72, 81, 82,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate no-visual slot/quality recurrent anatomy GAT.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-data-path", default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--geom-dim", type=int, default=37)
    parser.add_argument("--edge-attr-dim", type=int, default=22)
    parser.add_argument("--relabel-margin", type=float, default=0.025)
    parser.add_argument("--relabel-min-prob", type=float, default=0.12)
    parser.add_argument("--relabel-keep-prob-threshold", type=float, default=0.42)
    parser.add_argument("--relabel-structure-tolerance", type=float, default=0.20)
    parser.add_argument("--struct-candidate-weight", type=float, default=0.25)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.50)
    parser.add_argument("--duplicate-ios-threshold", type=float, default=0.70)
    parser.add_argument(
        "--score-calibration",
        choices=("raw", "gnn_delta", "ap_aware", "ap_aware_light", "ap_aware_domain"),
        default="ap_aware_domain",
        help=(
            "raw keeps detector scores except soft suppression; gnn_delta applies the older light GNN probability multiplier; "
            "ap_aware uses the learned score_delta head for COCO ranking calibration; "
            "ap_aware_light uses a smaller learned calibration to preserve detector AP; "
            "ap_aware_domain applies full calibration only when the graph shows external-domain score-shift evidence."
        ),
    )
    parser.add_argument("--ap-aware-score-weight", type=float, default=0.18)
    parser.add_argument("--ap-aware-prob-weight", type=float, default=0.04)
    parser.add_argument("--ap-aware-min-multiplier", type=float, default=0.72)
    parser.add_argument("--ap-aware-max-multiplier", type=float, default=1.12)
    parser.add_argument("--ap-aware-light-score-weight", type=float, default=0.06)
    parser.add_argument("--ap-aware-light-prob-weight", type=float, default=0.02)
    parser.add_argument("--ap-aware-light-min-multiplier", type=float, default=0.90)
    parser.add_argument("--ap-aware-light-max-multiplier", type=float, default=1.06)
    parser.add_argument("--ap-aware-domain-mean-threshold", type=float, default=-0.025)
    parser.add_argument("--ap-aware-domain-low-delta-threshold", type=float, default=-0.08)
    parser.add_argument("--ap-aware-domain-low-frac-threshold", type=float, default=0.28)
    parser.add_argument("--ap-aware-gate-delta-threshold", type=float, default=0.06)
    parser.add_argument("--ap-aware-gate-raw-score-threshold", type=float, default=0.80)
    parser.add_argument("--duplicate-score-weight", type=float, default=0.18)
    parser.add_argument(
        "--duplicate-action",
        choices=("low_score", "background"),
        default="low_score",
        help=(
            "How to suppress duplicates after one-to-one slot selection. "
            "low_score keeps the FDI label for COCO ranking and lowers score below the diagnostic threshold; "
            "background reproduces the old hard 48 suppression."
        ),
    )
    parser.add_argument("--duplicate-score-cap", type=float, default=0.19)
    parser.add_argument("--duplicate-score-multiplier", type=float, default=0.35)
    parser.add_argument("--disable-background-soft-suppression", action="store_true")
    parser.add_argument("--background-prob-threshold", type=float, default=0.40)
    parser.add_argument("--background-detector-score-threshold", type=float, default=0.40)
    parser.add_argument("--background-score-cap", type=float, default=0.19)
    parser.add_argument("--background-score-multiplier", type=float, default=0.35)
    parser.add_argument("--disable-cross-dentition-guard", action="store_true")
    parser.add_argument("--cross-dentition-guard-score-threshold", type=float, default=0.75)
    parser.add_argument("--high-score-adjacent-threshold", type=float, default=0.85)
    parser.add_argument("--drop-relabel-prob-threshold", type=float, default=0.55)
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention tensors. This forces batch_size=1 for the prediction export pass.",
    )
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, confusion_set):
    overall_acc = accuracy_score(y_true, y_pred)
    mask = np.isin(y_true, list(confusion_set))
    confusion_acc = (y_true[mask] == y_pred[mask]).sum() / mask.sum() if mask.sum() else 0.0
    return overall_acc, confusion_acc


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    labels = range(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_prior_labels(graph):
    if hasattr(graph, "pred_labels_raw"):
        return graph.pred_labels_raw.long().clamp(0, BACKGROUND_IDX)
    return graph.x_prior[:, :N_CLASSES].argmax(dim=1).long().clamp(0, BACKGROUND_IDX)


def get_detector_scores(graph):
    if hasattr(graph, "pred_scores_raw"):
        return graph.pred_scores_raw.float().clamp(0, 1)
    return ((graph.x_prior[:, -1].float() + 1.0) / 2.0).clamp(0, 1)


def get_raw_boxes(graph):
    if hasattr(graph, "pred_bboxes_raw"):
        return graph.pred_bboxes_raw.float()
    return graph.pos.float()


def get_dentition_groups(labels):
    lookup = torch.as_tensor(DENTITION_GROUPS, dtype=torch.long, device=labels.device)
    return lookup[labels.clamp(0, BACKGROUND_IDX)]


def get_jaw_groups(labels):
    lookup = torch.as_tensor(JAW_GROUPS, dtype=torch.long, device=labels.device)
    return lookup[labels.clamp(0, BACKGROUND_IDX)]


def get_side_groups(labels):
    lookup = torch.as_tensor(SIDE_GROUPS, dtype=torch.long, device=labels.device)
    return lookup[labels.clamp(0, BACKGROUND_IDX)]


def get_tooth_index(labels):
    lookup = torch.as_tensor(TOOTH_INDEX, dtype=torch.long, device=labels.device)
    return lookup[labels.clamp(0, BACKGROUND_IDX)]


def adjacent_or_same_structure(a, b):
    return (
        (get_dentition_groups(a) == get_dentition_groups(b))
        & (get_jaw_groups(a) == get_jaw_groups(b))
        & (get_side_groups(a) == get_side_groups(b))
        & ((get_tooth_index(a) - get_tooth_index(b)).abs() <= 1)
    )


def get_scalar(value, default=-1):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].detach().cpu().item())
    return int(value)


def box_iou_one_to_many(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return inter / (area_box + area_boxes - inter).clamp_min(1e-6)


def box_ios_one_to_many(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    smaller = torch.minimum(area_box.expand_as(area_boxes), area_boxes)
    return inter / smaller.clamp_min(1e-6)


def step_to_logits(step_output):
    slot_logits = step_output["slot_logits"]
    quality_logits = step_output["quality_logits"]
    keep_logit = quality_logits[:, 1:2]
    drop_logit = quality_logits[:, 0:1]
    class_logits = torch.cat([slot_logits + keep_logit, drop_logit], dim=1)
    return slot_logits, quality_logits, class_logits


def unpack_last_step(model_output):
    all_step_outputs = model_output[0] if isinstance(model_output, tuple) else model_output
    return all_step_outputs[-1] if isinstance(all_step_outputs, list) else all_step_outputs


def conservative_relabel(
    class_logits,
    quality_logits,
    raw_labels,
    detector_scores,
    slot_prior,
    relabel_margin,
    relabel_min_prob,
    relabel_keep_prob_threshold,
    relabel_structure_tolerance,
    struct_candidate_weight,
    high_score_adjacent_threshold,
    drop_relabel_prob_threshold,
):
    base_probs = F.softmax(class_logits, dim=1)
    quality_probs = F.softmax(quality_logits, dim=1)
    raw_labels = raw_labels.to(class_logits.device).clamp(0, BACKGROUND_IDX)
    candidate_logits = class_logits.clone()
    if slot_prior is not None and struct_candidate_weight > 0:
        candidate_logits[:, :BACKGROUND_IDX] = (
            candidate_logits[:, :BACKGROUND_IDX]
            + float(struct_candidate_weight) * slot_prior.to(class_logits.device)
        )
    candidate_probs = F.softmax(candidate_logits, dim=1)
    best_tooth_probs, best_tooth_labels = candidate_probs[:, :BACKGROUND_IDX].max(dim=1)
    raw_probs = candidate_probs.gather(1, raw_labels.unsqueeze(1)).squeeze(1)
    keep_probs = quality_probs[:, 1]
    drop_probs = quality_probs[:, 0]
    if slot_prior is None:
        structure_ok = torch.ones_like(best_tooth_probs, dtype=torch.bool)
        raw_struct = torch.zeros_like(best_tooth_probs)
        new_struct = torch.zeros_like(best_tooth_probs)
    else:
        slot_prior = slot_prior.to(class_logits.device)
        raw_struct = slot_prior.gather(1, raw_labels.clamp(0, BACKGROUND_IDX - 1).unsqueeze(1)).squeeze(1)
        new_struct = slot_prior.gather(1, best_tooth_labels.unsqueeze(1)).squeeze(1)
        structure_ok = (new_struct + relabel_structure_tolerance >= raw_struct) | adjacent_or_same_structure(raw_labels, best_tooth_labels)
    adjacent_ok = adjacent_or_same_structure(raw_labels, best_tooth_labels)
    high_score_protect = (detector_scores >= 0.75) & (raw_struct >= 0.25)
    strict_high_protect = detector_scores >= float(high_score_adjacent_threshold)
    drop_suppressed = (drop_probs >= float(drop_relabel_prob_threshold)) & (detector_scores <= 0.40)
    relabel_mask = (
        (best_tooth_labels != raw_labels)
        & (best_tooth_probs >= relabel_min_prob)
        & ((best_tooth_probs - raw_probs) >= relabel_margin)
        & (keep_probs >= relabel_keep_prob_threshold)
        & structure_ok
        & (~high_score_protect | adjacent_ok)
        & (~strict_high_protect | adjacent_ok)
        & (~drop_suppressed)
    )
    final_labels = raw_labels.clone()
    final_labels[relabel_mask] = best_tooth_labels[relabel_mask]
    return final_labels, candidate_probs, base_probs, best_tooth_labels, relabel_mask


def duplicate_cleanup(
    labels,
    boxes,
    detector_scores,
    class_probs,
    duplicate_iou_threshold,
    duplicate_ios_threshold,
    duplicate_action="low_score",
    quality_probs=None,
    slot_prior=None,
    score_delta=None,
    score_delta_weight=0.18,
):
    labels = labels.clone()
    duplicate_mask = torch.zeros_like(labels, dtype=torch.bool)
    if labels.numel() <= 1:
        return labels, duplicate_mask

    safe_labels = labels.clamp(0, BACKGROUND_IDX)
    label_probs = class_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    keep_score = detector_scores + 0.20 * label_probs
    if quality_probs is not None:
        keep_score = keep_score + 0.25 * quality_probs[:, 1]
    if slot_prior is not None:
        slot_prior = slot_prior.to(labels.device)
        slot_score = slot_prior.gather(1, safe_labels.clamp(0, BACKGROUND_IDX - 1).unsqueeze(1)).squeeze(1)
        keep_score = keep_score + 0.12 * slot_score.clamp(-2.0, 2.0)
    if score_delta is not None and score_delta_weight != 0:
        keep_score = keep_score + float(score_delta_weight) * torch.tanh(score_delta.to(labels.device)).clamp(-1.0, 1.0)

    for cls_idx in range(BACKGROUND_IDX):
        cls_indices = torch.where(labels == cls_idx)[0]
        if cls_indices.numel() <= 1:
            continue
        order = cls_indices[torch.argsort(keep_score[cls_indices], descending=True)]
        kept = []
        for idx in order:
            if not kept:
                kept.append(idx)
                continue
            kept_tensor = torch.stack(kept)
            ious = box_iou_one_to_many(boxes[idx], boxes[kept_tensor])
            ios = box_ios_one_to_many(boxes[idx], boxes[kept_tensor])
            if bool(((ious >= duplicate_iou_threshold) | (ios >= duplicate_ios_threshold)).any()):
                if duplicate_action == "background":
                    labels[idx] = BACKGROUND_IDX
                duplicate_mask[idx] = True
            else:
                kept.append(idx)
    return labels, duplicate_mask


def apply_cross_dentition_guard(
    labels,
    raw_labels,
    detector_scores,
    relabel_mask,
    enabled=True,
    score_threshold=0.80,
):
    labels = labels.clone()
    guard_mask = torch.zeros_like(labels, dtype=torch.bool)
    if not enabled or labels.numel() == 0:
        return labels, relabel_mask, guard_mask

    raw_dentition = get_dentition_groups(raw_labels)
    pred_dentition = get_dentition_groups(labels)
    guard_mask = (
        relabel_mask
        & (raw_labels != BACKGROUND_IDX)
        & (labels != BACKGROUND_IDX)
        & (raw_dentition != pred_dentition)
        & (detector_scores >= float(score_threshold))
    )
    if guard_mask.any():
        labels[guard_mask] = raw_labels[guard_mask]
        relabel_mask = relabel_mask.clone()
        relabel_mask[guard_mask] = False
    return labels, relabel_mask, guard_mask


def calibrate_scores(
    detector_scores,
    final_labels,
    class_probs,
    score_delta=None,
    duplicate_mask=None,
    score_calibration="raw",
    duplicate_action="low_score",
    duplicate_score_cap=0.19,
    duplicate_score_multiplier=0.35,
    background_soft_suppression=True,
    background_prob_threshold=0.40,
    background_detector_score_threshold=0.40,
    background_score_cap=0.19,
    background_score_multiplier=0.35,
    ap_score_weight=0.18,
    ap_prob_weight=0.04,
    ap_min_multiplier=0.72,
    ap_max_multiplier=1.12,
    ap_light_score_weight=0.06,
    ap_light_prob_weight=0.02,
    ap_light_min_multiplier=0.90,
    ap_light_max_multiplier=1.06,
    domain_mean_threshold=-0.025,
    domain_low_delta_threshold=-0.08,
    domain_low_frac_threshold=0.28,
    gate_delta_threshold=0.06,
    gate_raw_score_threshold=0.80,
):
    safe_labels = final_labels.clamp(0, BACKGROUND_IDX)
    label_probs = class_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    bg_probs = class_probs[:, BACKGROUND_IDX]

    if score_calibration == "ap_aware" and score_delta is not None:
        learned_delta = torch.tanh(score_delta).clamp(-1.0, 1.0)
        prob_delta = (label_probs - bg_probs).clamp(-1.0, 1.0)
        multiplier = (
            1.0
            + float(ap_score_weight) * learned_delta
            + float(ap_prob_weight) * prob_delta
        ).clamp(float(ap_min_multiplier), float(ap_max_multiplier))
        calibrated = (detector_scores * multiplier).clamp(0, 1)
    elif score_calibration == "ap_aware_light" and score_delta is not None:
        learned_delta = torch.tanh(score_delta).clamp(-1.0, 1.0)
        prob_delta = (label_probs - bg_probs).clamp(-1.0, 1.0)
        multiplier = (
            1.0
            + float(ap_light_score_weight) * learned_delta
            + float(ap_light_prob_weight) * prob_delta
        ).clamp(float(ap_light_min_multiplier), float(ap_light_max_multiplier))
        calibrated = (detector_scores * multiplier).clamp(0, 1)
    elif score_calibration == "ap_aware_domain" and score_delta is not None:
        learned_delta = torch.tanh(score_delta).clamp(-1.0, 1.0)
        prob_delta = (label_probs - bg_probs).clamp(-1.0, 1.0)
        low_delta_frac = (learned_delta <= float(domain_low_delta_threshold)).float().mean()
        domain_shifted = (
            learned_delta.mean() <= float(domain_mean_threshold)
            or low_delta_frac >= float(domain_low_frac_threshold)
        )
        if bool(domain_shifted):
            multiplier = (
                1.0
                + float(ap_score_weight) * learned_delta
                + float(ap_prob_weight) * prob_delta
            ).clamp(float(ap_min_multiplier), float(ap_max_multiplier))
        else:
            raw_multiplier = (
                1.0
                + float(ap_light_score_weight) * learned_delta
                + float(ap_light_prob_weight) * prob_delta
            ).clamp(float(ap_light_min_multiplier), float(ap_light_max_multiplier))
            evidence_mask = (
                (learned_delta.abs() >= float(gate_delta_threshold))
                | (detector_scores <= float(gate_raw_score_threshold))
                | (bg_probs >= float(background_prob_threshold))
            )
            multiplier = torch.ones_like(raw_multiplier)
            multiplier[evidence_mask] = raw_multiplier[evidence_mask]
        calibrated = (detector_scores * multiplier).clamp(0, 1)
    elif score_calibration == "gnn_delta":
        delta = (label_probs - bg_probs).clamp(-1.0, 1.0)
        multiplier = (1.0 + 0.12 * delta).clamp(0.88, 1.12)
        calibrated = (detector_scores * multiplier).clamp(0, 1)
    else:
        calibrated = detector_scores.clone().clamp(0, 1)

    calibrated[final_labels == BACKGROUND_IDX] = detector_scores[final_labels == BACKGROUND_IDX]
    if duplicate_mask is not None and duplicate_action == "low_score" and duplicate_mask.any():
        capped = torch.full_like(calibrated[duplicate_mask], float(duplicate_score_cap))
        softened = calibrated[duplicate_mask] * float(duplicate_score_multiplier)
        calibrated[duplicate_mask] = torch.minimum(softened, capped).clamp(0, 1)

    background_mask = torch.zeros_like(final_labels, dtype=torch.bool)
    if background_soft_suppression:
        background_mask = (
            (bg_probs >= float(background_prob_threshold))
            & (detector_scores <= float(background_detector_score_threshold))
            & (final_labels != BACKGROUND_IDX)
        )
        if duplicate_mask is not None:
            background_mask = background_mask & (~duplicate_mask)
        if background_mask.any():
            capped = torch.full_like(calibrated[background_mask], float(background_score_cap))
            softened = calibrated[background_mask] * float(background_score_multiplier)
            calibrated[background_mask] = torch.minimum(softened, capped).clamp(0, 1)

    return calibrated, background_mask


def postprocess_graph(step_output, graph, args):
    _, quality_logits, class_logits = step_to_logits(step_output)
    raw_labels = get_prior_labels(graph).to(class_logits.device)
    detector_scores = get_detector_scores(graph).to(class_logits.device)
    boxes = get_raw_boxes(graph).to(class_logits.device)
    slot_prior = getattr(graph, "x_slot_prior", None)
    quality_probs = F.softmax(quality_logits, dim=1)
    relabeled, class_probs, base_probs, best_tooth_labels, relabel_mask = conservative_relabel(
        class_logits,
        quality_logits,
        raw_labels,
        detector_scores,
        slot_prior,
        args.relabel_margin,
        args.relabel_min_prob,
        args.relabel_keep_prob_threshold,
        args.relabel_structure_tolerance,
        args.struct_candidate_weight,
        args.high_score_adjacent_threshold,
        args.drop_relabel_prob_threshold,
    )
    postprocessed, duplicate_mask = duplicate_cleanup(
        labels=relabeled,
        boxes=boxes,
        detector_scores=detector_scores,
        class_probs=class_probs,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        duplicate_ios_threshold=args.duplicate_ios_threshold,
        duplicate_action=args.duplicate_action,
        quality_probs=quality_probs,
        slot_prior=slot_prior.to(class_logits.device) if slot_prior is not None else None,
        score_delta=step_output.get("score_delta"),
        score_delta_weight=args.duplicate_score_weight,
    )
    postprocessed, relabel_mask, cross_dentition_guard_mask = apply_cross_dentition_guard(
        postprocessed,
        raw_labels,
        detector_scores,
        relabel_mask,
        not args.disable_cross_dentition_guard,
        args.cross_dentition_guard_score_threshold,
    )
    calibrated_scores, background_soft_mask = calibrate_scores(
        detector_scores,
        postprocessed,
        base_probs,
        step_output.get("score_delta"),
        duplicate_mask,
        args.score_calibration,
        args.duplicate_action,
        args.duplicate_score_cap,
        args.duplicate_score_multiplier,
        not args.disable_background_soft_suppression,
        args.background_prob_threshold,
        args.background_detector_score_threshold,
        args.background_score_cap,
        args.background_score_multiplier,
        args.ap_aware_score_weight,
        args.ap_aware_prob_weight,
        args.ap_aware_min_multiplier,
        args.ap_aware_max_multiplier,
        args.ap_aware_light_score_weight,
        args.ap_aware_light_prob_weight,
        args.ap_aware_light_min_multiplier,
        args.ap_aware_light_max_multiplier,
        args.ap_aware_domain_mean_threshold,
        args.ap_aware_domain_low_delta_threshold,
        args.ap_aware_domain_low_frac_threshold,
        args.ap_aware_gate_delta_threshold,
        args.ap_aware_gate_raw_score_threshold,
    )
    return {
        "pred_original": raw_labels.detach().cpu(),
        "pred_model_best_tooth": best_tooth_labels.detach().cpu(),
        "pred_postprocessed": postprocessed.detach().cpu(),
        "pred_scores_gnn": calibrated_scores.detach().cpu(),
        "gnn_probs": class_probs.detach().cpu(),
        "gnn_logits": class_logits.detach().cpu(),
        "relabel_mask": relabel_mask.detach().cpu(),
        "duplicate_suppressed_mask": duplicate_mask.detach().cpu(),
        "background_soft_suppressed_mask": background_soft_mask.detach().cpu(),
        "cross_dentition_guard_mask": cross_dentition_guard_mask.detach().cpu(),
    }


@torch.no_grad()
def evaluate_aggregate(model, loader, device, args):
    model.eval()
    all_y_true, all_pred_original, all_pred_gnn = [], [], []
    relabel_total = 0
    duplicate_suppressed_total = 0
    background_soft_suppressed_total = 0
    cross_dentition_guard_total = 0
    total_inference_time = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        out = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_inference_time += time.time() - start
        total_graphs += batch.num_graphs

        node_offset = 0
        last_steps = out[-1]
        for graph in batch.to_data_list():
            num_nodes = graph.num_nodes
            single_step = {
                "slot_logits": last_steps["slot_logits"][node_offset:node_offset + num_nodes],
                "quality_logits": last_steps["quality_logits"][node_offset:node_offset + num_nodes],
            }
            if "score_delta" in last_steps:
                single_step["score_delta"] = last_steps["score_delta"][node_offset:node_offset + num_nodes]
            processed = postprocess_graph(single_step, graph, args)
            all_y_true.append(graph.y.long().cpu().numpy())
            all_pred_original.append(processed["pred_original"].numpy())
            all_pred_gnn.append(processed["pred_postprocessed"].numpy())
            relabel_total += int(processed["relabel_mask"].sum())
            duplicate_suppressed_total += int(processed["duplicate_suppressed_mask"].sum())
            background_soft_suppressed_total += int(processed["background_soft_suppressed_mask"].sum())
            cross_dentition_guard_total += int(processed["cross_dentition_guard_mask"].sum())
            node_offset += num_nodes

    avg_latency_ms = (total_inference_time / total_graphs) * 1000 if total_graphs else 0.0
    fps = total_graphs / total_inference_time if total_inference_time > 0 else 0.0
    logging.info("inference speed")
    logging.info(f"total={total_inference_time:.4f}s, graphs={total_graphs}, latency={avg_latency_ms:.4f}ms, fps={fps:.2f}")
    logging.info(
        f"relabel nodes={relabel_total}, duplicate suppressed nodes={duplicate_suppressed_total}, "
        f"background soft-suppressed nodes={background_soft_suppressed_total}, "
        f"cross-dentition guarded nodes={cross_dentition_guard_total}"
    )
    return (
        np.concatenate(all_y_true),
        np.concatenate(all_pred_original),
        np.concatenate(all_pred_gnn),
        fps,
        avg_latency_ms,
    )


@torch.no_grad()
def generate_per_graph_predictions(model, test_data_list, device, save_path, args):
    export_batch_size = 1 if args.save_attention else args.batch_size
    loader = DataLoader(test_data_list, batch_size=export_batch_size, shuffle=False)
    model.eval()
    all_graph_results = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, return_att=args.save_attention)
        step_output = unpack_last_step(out)
        att_weights, edge_index, edge_type = None, None, None
        if args.save_attention and batch.num_graphs == 1 and isinstance(out, tuple) and len(out) >= 4:
            _, att_weights, edge_index, edge_type = out
        node_offset = 0
        for graph in batch.to_data_list():
            num_nodes = graph.num_nodes
            single_step = {
                "slot_logits": step_output["slot_logits"][node_offset:node_offset + num_nodes],
                "quality_logits": step_output["quality_logits"][node_offset:node_offset + num_nodes],
            }
            if "score_delta" in step_output:
                single_step["score_delta"] = step_output["score_delta"][node_offset:node_offset + num_nodes]
            processed = postprocess_graph(single_step, graph, args)
            all_graph_results.append({
                "img_id": get_scalar(getattr(graph, "img_id", None)),
                "img_path": graph.img_path[0] if isinstance(graph.img_path, list) else graph.img_path,
                "y_true": graph.y.long().cpu(),
                "slot_y": graph.slot_y.long().cpu() if hasattr(graph, "slot_y") else None,
                "quality_y": graph.quality_y.long().cpu() if hasattr(graph, "quality_y") else None,
                "pred_original": processed["pred_original"],
                "pred_gnn": processed["pred_postprocessed"],
                "pred_postprocessed": processed["pred_postprocessed"],
                "pred_model_best_tooth": processed["pred_model_best_tooth"],
                "pred_scores_raw": get_detector_scores(graph).cpu(),
                "pred_scores_gnn": processed["pred_scores_gnn"],
                "gnn_logits": processed["gnn_logits"],
                "gnn_probs": processed["gnn_probs"],
                "score_delta": single_step.get("score_delta", torch.empty((num_nodes,), device=step_output["slot_logits"].device)).detach().cpu(),
                "fusion_weights": step_output.get("fusion_weights", torch.empty((num_nodes, 3), device=step_output["slot_logits"].device))[node_offset:node_offset + num_nodes].detach().cpu(),
                "relabel_mask": processed["relabel_mask"],
                "duplicate_suppressed_mask": processed["duplicate_suppressed_mask"],
                "background_soft_suppressed_mask": processed["background_soft_suppressed_mask"],
                "cross_dentition_guard_mask": processed["cross_dentition_guard_mask"],
                "edge_index": edge_index.cpu().numpy() if edge_index is not None else None,
                "att_weights": att_weights.cpu().numpy() if att_weights is not None else None,
                "edge_types": edge_type.cpu().numpy() if edge_type is not None else None,
                "raw_boxes": get_raw_boxes(graph).cpu().numpy(),
            })
            node_offset += num_nodes
    torch.save(all_graph_results, save_path)
    logging.info(f"per-graph predictions saved to {save_path}")


def build_model(args):
    return SlotQualityRecurrentAnatomyGAT(
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
    ).to(DEVICE)


def main():
    args = parse_args()
    results_save_dir = os.path.dirname(args.model_path) or "."
    os.makedirs(results_save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(results_save_dir, "evaluation_recurrent.log")), logging.StreamHandler()],
    )

    test_data_list = torch.load(args.test_data_path, map_location="cpu", weights_only=False)
    if test_data_list:
        args.geom_dim = int(test_data_list[0].x_geom.size(1))
        args.edge_attr_dim = int(test_data_list[0].edge_attr_overlap.size(1))
    logging.info(
        f"model=SlotQualityRecurrentAnatomyGAT, T={args.num_iterations}, use_visual=False, "
        f"spatial_only=False, geom_dim={args.geom_dim}, edge_attr_dim={args.edge_attr_dim}"
    )
    logging.info(f"test_data={args.test_data_path}, graphs={len(test_data_list)}")
    logging.info(
        f"postprocess: relabel_margin={args.relabel_margin}, relabel_min_prob={args.relabel_min_prob}, "
        f"relabel_keep_prob_threshold={args.relabel_keep_prob_threshold}, "
        f"relabel_structure_tolerance={args.relabel_structure_tolerance}, "
        f"struct_candidate_weight={args.struct_candidate_weight}, "
        f"duplicate_iou={args.duplicate_iou_threshold}, duplicate_ios={args.duplicate_ios_threshold}, "
        f"duplicate_action={args.duplicate_action}, duplicate_score_cap={args.duplicate_score_cap}, "
        f"score_calibration={args.score_calibration}, "
        f"ap_aware=({args.ap_aware_score_weight}/{args.ap_aware_prob_weight}/"
        f"{args.ap_aware_min_multiplier}-{args.ap_aware_max_multiplier}), "
        f"ap_aware_light=({args.ap_aware_light_score_weight}/{args.ap_aware_light_prob_weight}/"
        f"{args.ap_aware_light_min_multiplier}-{args.ap_aware_light_max_multiplier}), "
        f"ap_aware_domain=(mean<={args.ap_aware_domain_mean_threshold}, "
        f"low_delta<={args.ap_aware_domain_low_delta_threshold}, "
        f"low_frac>={args.ap_aware_domain_low_frac_threshold}, "
        f"gate_delta>={args.ap_aware_gate_delta_threshold}, "
        f"gate_raw<={args.ap_aware_gate_raw_score_threshold}), "
        f"duplicate_score_weight={args.duplicate_score_weight}, "
        f"background_soft_suppression={not args.disable_background_soft_suppression}, "
        f"background_prob_threshold={args.background_prob_threshold}, "
        f"background_detector_score_threshold={args.background_detector_score_threshold}, "
        f"cross_dentition_guard={not args.disable_cross_dentition_guard}, "
        f"cross_dentition_guard_score_threshold={args.cross_dentition_guard_score_threshold}"
    )

    if not os.path.exists(args.model_path):
        logging.error(f"model file not found: {args.model_path}")
        return

    model = build_model(args)
    state = torch.load(args.model_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    load_result = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "slot_fusion_floor",
        "score_delta_head.0.weight",
        "score_delta_head.0.bias",
        "score_delta_head.3.weight",
        "score_delta_head.3.bias",
    }
    missing = [key for key in load_result.missing_keys if key not in allowed_missing]
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. missing={missing}, unexpected={load_result.unexpected_keys}"
        )
    if load_result.missing_keys:
        logging.warning(f"ignored missing non-parameter keys: {load_result.missing_keys}")
    logging.info(f"loaded model: {args.model_path}")

    loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
    y_true, pred_original, pred_gnn, fps, latency = evaluate_aggregate(model, loader, DEVICE, args)

    metrics_orig = calculate_metrics(y_true, pred_original, CONFUSION_SET_CLASSES)
    metrics_gnn = calculate_metrics(y_true, pred_gnn, CONFUSION_SET_CLASSES)
    logging.info("=" * 70)
    logging.info(f"{'metric':<24} | {'Detector':<15} | {'GNN corrected':<15}")
    logging.info("-" * 70)
    logging.info(f"{'overall accuracy':<24} | {metrics_orig[0]:.4f}          | {metrics_gnn[0]:.4f}")
    logging.info(f"{'confusion accuracy':<24} | {metrics_orig[1]:.4f}          | {metrics_gnn[1]:.4f}")
    logging.info(f"{'fps':<24} | {'-':<15} | {fps:.2f}")
    logging.info(f"{'latency ms':<24} | {'-':<15} | {latency:.2f}")
    logging.info("=" * 70)

    class_names = [str(i) for i in range(N_CLASSES)]
    report_gnn = classification_report(y_true, pred_gnn, target_names=class_names, zero_division=0)
    with open(os.path.join(results_save_dir, "classification_report_recurrent.txt"), "w") as f:
        f.write(report_gnn)
    logging.info("classification report:\n" + report_gnn)
    plot_confusion_matrix(
        y_true,
        pred_gnn,
        class_names,
        f"Confusion Matrix - SlotQuality RecurrentGAT (T={args.num_iterations})",
        os.path.join(results_save_dir, "cm_recurrent_gat.png"),
    )

    save_pt_path = os.path.join(results_save_dir, args.output_name)
    generate_per_graph_predictions(model, test_data_list, DEVICE, save_pt_path, args)
    logging.info("evaluation finished")


if __name__ == "__main__":
    main()
