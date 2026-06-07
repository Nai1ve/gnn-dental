from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops
from pycocotools.coco import COCO
from torch_geometric.data import Data
from tqdm import tqdm


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
BACKGROUND = len(CLASS_NAMES)
N_CLASSES = BACKGROUND + 1
EPS = 1e-6
DENTITION_GROUPS = np.asarray([1 if int(name) // 10 >= 5 else 0 for name in CLASS_NAMES] + [2], dtype=np.int64)
JAW_GROUPS = np.asarray([0 if int(name) // 10 in (1, 2, 5, 6) else 1 for name in CLASS_NAMES] + [2], dtype=np.int64)
SIDE_GROUPS = np.asarray([0 if int(name) // 10 in (1, 4, 5, 8) else 1 for name in CLASS_NAMES] + [2], dtype=np.int64)
TOOTH_ORDER = np.asarray([
    (int(name) % 10 - 1) / (4.0 if int(name) // 10 >= 5 else 6.0)
    for name in CLASS_NAMES
] + [0.0], dtype=np.float32)
TOOTH_INDEX = np.asarray([int(name) % 10 for name in CLASS_NAMES] + [0], dtype=np.int64)
EXPECTED_JAW_RANK = np.asarray([
    (1.0 if int(name) // 10 in (2, 3, 6, 7) else -1.0)
    * ((int(name) % 10 - 1) / (4.0 if int(name) // 10 >= 5 else 6.0))
    for name in CLASS_NAMES
] + [0.0], dtype=np.float32)


def is_missing_feature(value) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray) and value.shape == ():
        try:
            return value.item() is None
        except ValueError:
            return False
    return False


def load_raw_features(path: Path):
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def build_fallback_class_probs(pred_labels, pred_scores):
    labels = to_numpy(pred_labels, np.int64)
    scores = np.clip(to_numpy(pred_scores, np.float32), 0.0, 1.0)
    probs = np.zeros((len(labels), N_CLASSES), dtype=np.float32)
    valid = (labels >= 0) & (labels < BACKGROUND)
    rows = np.arange(len(labels))[valid]
    probs[rows, labels[valid]] = scores[valid]
    probs[:, BACKGROUND] = 1.0 - scores
    return probs


def normalize_stem_for_match(stem: str) -> str:
    normalized = stem.replace(" ", "")
    changed = True
    while changed:
        changed = False
        for prefix in ("qy", "dsy"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
    if normalized.isdigit():
        normalized = normalized.lstrip("0") or "0"
    return normalized


def filename_aliases(filename) -> set[str]:
    if not filename:
        return set()
    base = os.path.basename(str(filename))
    stem, ext = os.path.splitext(base)
    stems = {stem, stem.replace(" ", ""), normalize_stem_for_match(stem)}
    if stem.isdigit():
        stems.add(stem.lstrip("0") or "0")
    exts = {ext, ext.lower(), ext.upper(), ""}
    aliases = {base}
    for s in stems:
        for e in exts:
            aliases.add(f"{s}{e}")
    return {a for a in aliases if a}


def build_filename_to_id_map(coco_gt: COCO) -> dict[str, int]:
    alias_to_id = {}
    ambiguous = set()

    def add(alias, img_id):
        if alias in ambiguous:
            return
        old = alias_to_id.get(alias)
        if old is not None and old != img_id:
            del alias_to_id[alias]
            ambiguous.add(alias)
            return
        alias_to_id[alias] = img_id

    for img in coco_gt.dataset["images"]:
        for alias in filename_aliases(img.get("file_name")):
            add(alias, int(img["id"]))
    return alias_to_id


def resolve_img_id(raw_item: dict, filename_to_id: dict[str, int], coco_gt: COCO) -> Optional[int]:
    raw_img_id = raw_item.get("img_id")
    try:
        if isinstance(raw_img_id, torch.Tensor):
            raw_img_id = raw_img_id.item()
        if isinstance(raw_img_id, np.generic):
            raw_img_id = raw_img_id.item()
        raw_img_id = int(raw_img_id)
    except (TypeError, ValueError):
        raw_img_id = None

    if raw_img_id is not None and raw_img_id in coco_gt.imgs:
        return raw_img_id

    raw_name = raw_item.get("file_name") or raw_item.get("img_path")
    for alias in filename_aliases(raw_name):
        img_id = filename_to_id.get(alias)
        if img_id is not None:
            return img_id
    return None


def fourier_encode(x: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    x = x.unsqueeze(-1)
    freqs = torch.pow(2, torch.arange(num_bands, dtype=torch.float, device=x.device))
    x_freq = x * freqs * torch.pi
    return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)


def label_structure_features(labels: torch.Tensor, is_upper: torch.Tensor, is_left: torch.Tensor, rank_jaw: torch.Tensor):
    safe_labels = labels.clamp(0, BACKGROUND)
    device = labels.device
    dentition = torch.as_tensor(DENTITION_GROUPS, dtype=torch.long, device=device)[safe_labels]
    jaw = torch.as_tensor(JAW_GROUPS, dtype=torch.long, device=device)[safe_labels]
    side = torch.as_tensor(SIDE_GROUPS, dtype=torch.long, device=device)[safe_labels]
    tooth_order = torch.as_tensor(TOOTH_ORDER, dtype=torch.float, device=device)[safe_labels]
    expected_jaw_rank = torch.as_tensor(EXPECTED_JAW_RANK, dtype=torch.float, device=device)[safe_labels]

    raw_is_upper = jaw == 0
    raw_is_left = side == 0
    valid_tooth = safe_labels < BACKGROUND
    dentition_sign = torch.where(dentition == 1, torch.ones_like(tooth_order), -torch.ones_like(tooth_order))
    jaw_sign = torch.where(raw_is_upper, -torch.ones_like(tooth_order), torch.ones_like(tooth_order))
    side_sign = torch.where(raw_is_left, -torch.ones_like(tooth_order), torch.ones_like(tooth_order))
    jaw_match = ((raw_is_upper == is_upper) & valid_tooth).float()
    side_match = ((raw_is_left == is_left) & valid_tooth).float()
    order_rank_delta = (expected_jaw_rank - rank_jaw).clamp(-2.0, 2.0) / 2.0
    valid_float = valid_tooth.float()
    return torch.stack([
        dentition_sign,
        jaw_sign,
        side_sign,
        tooth_order * 2.0 - 1.0,
        jaw_match * 2.0 - 1.0,
        side_match * 2.0 - 1.0,
        order_rank_delta,
        valid_float,
    ], dim=1).float()


def calculate_iou_np(box_a, box_b) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + EPS)


def generate_assignments(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_labels: np.ndarray,
    active_iou_threshold: float,
    slot_iou_threshold: float,
):
    num_preds = len(pred_bboxes)
    legacy_y = np.full(num_preds, BACKGROUND, dtype=np.int64)
    slot_y = np.full(num_preds, -100, dtype=np.int64)
    quality_y = np.zeros(num_preds, dtype=np.int64)
    node_role = np.full(num_preds, 3, dtype=np.int64)
    best_gt_iou = np.zeros(num_preds, dtype=np.float32)

    if num_preds == 0 or len(gt_bboxes) == 0:
        return legacy_y, slot_y, quality_y, node_role, best_gt_iou

    iou_matrix = np.array([
        [calculate_iou_np(pred_box, gt_box) for gt_box in gt_bboxes]
        for pred_box in pred_bboxes
    ], dtype=np.float32)

    best_gt = iou_matrix.argmax(axis=1)
    best_gt_iou = iou_matrix[np.arange(num_preds), best_gt]
    slot_mask = best_gt_iou >= slot_iou_threshold
    slot_y[slot_mask] = gt_labels[best_gt[slot_mask]].astype(np.int64)

    sorted_pred_indices = np.argsort(pred_scores)[::-1]
    gt_matched = np.zeros(len(gt_bboxes), dtype=bool)
    for pred_idx in sorted_pred_indices:
        candidate_gt_order = np.argsort(iou_matrix[pred_idx])[::-1]
        for gt_idx in candidate_gt_order:
            if gt_matched[gt_idx]:
                continue
            if iou_matrix[pred_idx, gt_idx] < active_iou_threshold:
                break
            gt_matched[gt_idx] = True
            label = int(gt_labels[gt_idx])
            legacy_y[pred_idx] = label
            quality_y[pred_idx] = 1
            node_role[pred_idx] = 0
            if slot_y[pred_idx] == -100:
                slot_y[pred_idx] = label
            break

    duplicate_mask = (quality_y == 0) & (slot_y >= 0)
    node_role[duplicate_mask] = 1
    near_mask = (quality_y == 0) & (slot_y < 0) & (best_gt_iou >= 0.10)
    node_role[near_mask] = 2
    return legacy_y, slot_y, quality_y, node_role, best_gt_iou


def jaw_side_rank_features(centers: torch.Tensor, img_w: float, img_h: float):
    n = centers.size(0)
    device = centers.device
    x_mid = torch.median(centers[:, 0])
    y_mid = torch.median(centers[:, 1])
    is_upper = centers[:, 1] < y_mid
    is_left = centers[:, 0] < x_mid

    jaw_sign = torch.where(is_upper, -torch.ones(n, device=device), torch.ones(n, device=device))
    side_sign = torch.where(is_left, -torch.ones(n, device=device), torch.ones(n, device=device))

    rank_all = torch.zeros(n, device=device)
    rank_jaw = torch.zeros(n, device=device)
    rank_quadrant = torch.zeros(n, device=device)

    order_all = torch.argsort(centers[:, 0])
    if n > 1:
        rank_all[order_all] = torch.linspace(-1.0, 1.0, steps=n, device=device)

    for mask in (is_upper, ~is_upper):
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            continue
        order = idx[torch.argsort(centers[idx, 0])]
        rank_jaw[order] = torch.linspace(-1.0, 1.0, steps=idx.numel(), device=device)

    for jaw_mask in (is_upper, ~is_upper):
        for side_mask in (is_left, ~is_left):
            idx = torch.where(jaw_mask & side_mask)[0]
            if idx.numel() == 0:
                continue
            order = idx[torch.argsort(centers[idx, 0])]
            rank_quadrant[order] = torch.linspace(-1.0, 1.0, steps=idx.numel(), device=device)

    dist_mid_x = ((centers[:, 0] - x_mid) / max(img_w, EPS)).clamp(-1, 1)
    dist_mid_y = ((centers[:, 1] - y_mid) / max(img_h, EPS)).clamp(-1, 1)

    node_space = torch.stack([
        jaw_sign,
        side_sign,
        rank_all,
        rank_jaw,
        rank_quadrant,
        dist_mid_x,
        dist_mid_y,
    ], dim=1)
    return node_space, is_upper, is_left, rank_jaw


def build_multi_relation_graph(
    boxes: torch.Tensor,
    centers: torch.Tensor,
    iou_threshold: float,
    ios_threshold: float,
    k_arch: int,
    k_spatial: int,
    y_penalty: float,
    x_penalty: float,
):
    n = boxes.size(0)
    device = boxes.device
    empty = torch.empty((2, 0), dtype=torch.long, device=device)
    if n < 2:
        return empty, empty, empty, empty, torch.zeros((n, n), device=device), torch.zeros((n, n), device=device)

    iou = torchvision.ops.box_iou(boxes, boxes)
    area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    lt = torch.max(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    min_area = torch.minimum(area[:, None], area[None, :])
    ios = inter / min_area.clamp_min(EPS)

    is_overlap = (iou > iou_threshold) | (ios > ios_threshold)
    is_overlap.fill_diagonal_(False)
    edge_overlap = is_overlap.nonzero(as_tuple=False).t().contiguous()

    forbidden = is_overlap | torch.eye(n, dtype=torch.bool, device=device)

    scaled_arch = centers.clone()
    scaled_arch[:, 1] *= y_penalty
    arch_dist = torch.cdist(scaled_arch, scaled_arch, p=2)
    arch_dist[forbidden] = float("inf")
    edge_arch = topk_edges(arch_dist, k_arch)

    y_mid = torch.median(centers[:, 1])
    is_upper = centers[:, 1] < y_mid
    cross_jaw = (is_upper[:, None] & (~is_upper)[None, :]) | ((~is_upper)[:, None] & is_upper[None, :])
    scaled_vert = centers.clone()
    scaled_vert[:, 0] *= x_penalty
    vert_dist = torch.cdist(scaled_vert, scaled_vert, p=2)
    vert_dist[forbidden | cross_jaw] = float("inf")
    edge_vertical = topk_edges(vert_dist, 1)

    spatial_dist = torch.cdist(centers, centers, p=2)
    spatial_dist.fill_diagonal_(float("inf"))
    edge_spatial = topk_edges(spatial_dist, k_spatial)
    return edge_overlap, edge_arch, edge_vertical, edge_spatial, iou, ios


def topk_edges(dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
    n = dist_matrix.size(0)
    curr_k = min(k, n - 1)
    if curr_k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=dist_matrix.device)
    dists, indices = torch.topk(dist_matrix, k=curr_k, dim=1, largest=False)
    valid = dists != float("inf")
    sources = torch.arange(n, device=dist_matrix.device).unsqueeze(1).expand(n, curr_k)
    return torch.stack([sources[valid], indices[valid]], dim=0)


def compute_edge_attributes(
    edge_index: torch.Tensor,
    boxes: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
    img_w: float,
    img_h: float,
    iou_matrix: torch.Tensor,
    ios_matrix: torch.Tensor,
    is_upper: torch.Tensor,
    is_left: torch.Tensor,
    rank_jaw: torch.Tensor,
):
    if edge_index.numel() == 0:
        return torch.empty((0, 16), dtype=torch.float, device=boxes.device)
    src, dst = edge_index[0], edge_index[1]
    dx = (centers[dst, 0] - centers[src, 0]) / max(img_w, EPS)
    dy = (centers[dst, 1] - centers[src, 1]) / max(img_h, EPS)
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    w = (boxes[:, 2] - boxes[:, 0]).clamp_min(EPS)
    h = (boxes[:, 3] - boxes[:, 1]).clamp_min(EPS)
    log_w = torch.log(w[dst] / w[src])
    log_h = torch.log(h[dst] / h[src])
    same_jaw = (is_upper[src] == is_upper[dst]).float()
    same_side = (is_left[src] == is_left[dst]).float()
    rank_delta = (rank_jaw[dst] - rank_jaw[src]).clamp(-2.0, 2.0) / 2.0
    direction_sign = torch.sign(dx)
    safe_labels = labels.clamp(0, BACKGROUND)
    device = labels.device
    dentition = torch.as_tensor(DENTITION_GROUPS, dtype=torch.long, device=device)[safe_labels]
    label_jaw = torch.as_tensor(JAW_GROUPS, dtype=torch.long, device=device)[safe_labels]
    label_side = torch.as_tensor(SIDE_GROUPS, dtype=torch.long, device=device)[safe_labels]
    label_order = torch.as_tensor(TOOTH_ORDER, dtype=torch.float, device=device)[safe_labels]
    label_index = torch.as_tensor(TOOTH_INDEX, dtype=torch.long, device=device)[safe_labels]
    valid_pair = ((safe_labels[src] < BACKGROUND) & (safe_labels[dst] < BACKGROUND)).float()
    same_dentition_label = ((dentition[src] == dentition[dst]).float() * valid_pair)
    same_jaw_label = ((label_jaw[src] == label_jaw[dst]).float() * valid_pair)
    same_side_label = ((label_side[src] == label_side[dst]).float() * valid_pair)
    order_delta_label = ((label_order[dst] - label_order[src]).clamp(-1.0, 1.0) * valid_pair)
    adjacent_label = ((torch.abs(label_index[dst] - label_index[src]) == 1).float()
                      * same_dentition_label * same_jaw_label * same_side_label)
    return torch.stack([
        dx,
        dy,
        dist,
        log_w,
        log_h,
        iou_matrix[src, dst],
        ios_matrix[src, dst],
        same_jaw,
        same_side,
        rank_delta,
        direction_sign,
        same_dentition_label,
        same_jaw_label,
        same_side_label,
        order_delta_label,
        adjacent_label,
    ], dim=1).float()


def build_graph_from_features(
    raw_item: dict,
    gt_item: dict,
    min_score: float,
    active_iou_threshold: float,
    slot_iou_threshold: float,
    iou_threshold_graph: float,
    ios_threshold_graph: float,
    k_arch: int,
    k_spatial: int,
    y_penalty: float,
    visual_dim: int,
    force_fallback_prior: bool,
):
    pred_bboxes = to_numpy(raw_item["pred_bboxes"], np.float32)
    pred_scores = to_numpy(raw_item["pred_scores"], np.float32)
    pred_labels = to_numpy(raw_item["pred_labels"], np.int64)
    if len(pred_bboxes) == 0:
        return None

    legacy_y, slot_y, quality_y, node_role, best_gt_iou = generate_assignments(
        pred_bboxes,
        pred_scores,
        gt_item["gt_bboxes_np"],
        gt_item["gt_labels_np"],
        active_iou_threshold=active_iou_threshold,
        slot_iou_threshold=slot_iou_threshold,
    )

    score_mask = pred_scores >= min_score
    if int(score_mask.sum()) == 0:
        return None

    if force_fallback_prior or is_missing_feature(raw_item.get("pred_all_class_probs")):
        pred_all_class_probs = build_fallback_class_probs(pred_labels, pred_scores)
        used_fallback_prior = True
    else:
        pred_all_class_probs = to_numpy(raw_item["pred_all_class_probs"], np.float32)
        used_fallback_prior = False

    img_h, img_w = raw_item["ori_shape"][0], raw_item["ori_shape"][1]
    boxes = torch.tensor(pred_bboxes[score_mask], dtype=torch.float)
    scores = torch.tensor(pred_scores[score_mask], dtype=torch.float)
    labels = torch.tensor(pred_labels[score_mask], dtype=torch.long)
    y = torch.tensor(legacy_y[score_mask], dtype=torch.long)
    slot = torch.tensor(slot_y[score_mask], dtype=torch.long)
    quality = torch.tensor(quality_y[score_mask], dtype=torch.long)
    role = torch.tensor(node_role[score_mask], dtype=torch.long)
    best_iou = torch.tensor(best_gt_iou[score_mask], dtype=torch.float)
    class_probs = torch.tensor(pred_all_class_probs[score_mask], dtype=torch.float)

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = torch.stack([centers_x, centers_y], dim=1)

    centroid = centers.mean(dim=0)
    enc_dx = fourier_encode((centers_x - centroid[0]) / max(img_w, EPS), num_bands=4)
    enc_dy = fourier_encode((centers_y - centroid[1]) / max(img_h, EPS), num_bands=4)
    rel_coord = torch.cat([enc_dx, enc_dy], dim=1)
    node_space, is_upper, is_left, rank_jaw = jaw_side_rank_features(centers, img_w, img_h)
    label_space = label_structure_features(labels, is_upper, is_left, rank_jaw)

    geom_features = []
    for i in range(boxes.size(0)):
        box = boxes[i]
        w = (box[2] - box[0]).clamp_min(EPS)
        h = (box[3] - box[1]).clamp_min(EPS)
        base = torch.tensor([
            (centers_x[i] / img_w) * 2 - 1,
            (centers_y[i] / img_h) * 2 - 1,
            (w / img_w) * 2 - 1,
            (h / img_h) * 2 - 1,
            ((w * h) / (img_w * img_h)) * 2 - 1,
            math.log((w / h).item()),
        ], dtype=torch.float)
        geom_features.append(torch.cat([base, rel_coord[i], node_space[i], label_space[i]], dim=0))
    x_geom = torch.stack(geom_features).float()

    x_visual = torch.zeros((boxes.size(0), visual_dim), dtype=torch.float)
    x_prior = torch.cat([class_probs, (scores * 2 - 1).unsqueeze(1)], dim=1).float()

    edge_overlap, edge_arch, edge_vertical, edge_spatial, iou_matrix, ios_matrix = build_multi_relation_graph(
        boxes=boxes,
        centers=centers,
        iou_threshold=iou_threshold_graph,
        ios_threshold=ios_threshold_graph,
        k_arch=k_arch,
        k_spatial=k_spatial,
        y_penalty=y_penalty,
        x_penalty=5.0,
    )

    edge_attr_overlap = compute_edge_attributes(edge_overlap, boxes, centers, labels, img_w, img_h, iou_matrix, ios_matrix, is_upper, is_left, rank_jaw)
    edge_attr_arch = compute_edge_attributes(edge_arch, boxes, centers, labels, img_w, img_h, iou_matrix, ios_matrix, is_upper, is_left, rank_jaw)
    edge_attr_vertical = compute_edge_attributes(edge_vertical, boxes, centers, labels, img_w, img_h, iou_matrix, ios_matrix, is_upper, is_left, rank_jaw)
    edge_attr_spatial = compute_edge_attributes(edge_spatial, boxes, centers, labels, img_w, img_h, iou_matrix, ios_matrix, is_upper, is_left, rank_jaw)

    pos_normalized = torch.stack([
        (centers[:, 0] / img_w) * 2 - 1,
        (centers[:, 1] / img_h) * 2 - 1,
    ], dim=1).float()

    data = Data(
        x_visual=x_visual,
        x_geom=x_geom,
        x_prior=x_prior,
        edge_index_overlap=edge_overlap,
        edge_index_arch=edge_arch,
        edge_index_vertical=edge_vertical,
        edge_index_spatial=edge_spatial,
        edge_attr_overlap=edge_attr_overlap,
        edge_attr_arch=edge_attr_arch,
        edge_attr_vertical=edge_attr_vertical,
        edge_attr_spatial=edge_attr_spatial,
        y=y,
        slot_y=slot,
        quality_y=quality,
        node_role=role,
        best_gt_iou=best_iou,
        pos=pos_normalized,
        img_id=int(raw_item["img_id"]),
        img_path=raw_item["img_path"],
        ori_shape=torch.tensor(raw_item["ori_shape"]),
        pred_bboxes_raw=boxes,
        pred_scores_raw=scores,
        pred_labels_raw=labels,
        pred_label_dentition=torch.as_tensor(DENTITION_GROUPS, dtype=torch.long)[labels.clamp(0, BACKGROUND)],
        pred_label_jaw=torch.as_tensor(JAW_GROUPS, dtype=torch.long)[labels.clamp(0, BACKGROUND)],
        pred_label_side=torch.as_tensor(SIDE_GROUPS, dtype=torch.long)[labels.clamp(0, BACKGROUND)],
        pred_label_order=torch.as_tensor(TOOTH_ORDER, dtype=torch.float)[labels.clamp(0, BACKGROUND)],
        used_fallback_prior=used_fallback_prior,
        visual_missing=True,
    )
    return data


def standardize_raw_item(raw_item: dict, filename_to_id: dict[str, int], coco_gt: COCO) -> Optional[dict]:
    img_id = resolve_img_id(raw_item, filename_to_id, coco_gt)
    if img_id is None:
        return None
    img_info = coco_gt.loadImgs(img_id)[0]
    raw_item = dict(raw_item)
    raw_item["img_id"] = img_id
    raw_item["img_path"] = img_info["file_name"]
    if "ori_shape" not in raw_item or raw_item["ori_shape"] is None:
        raw_item["ori_shape"] = (img_info["height"], img_info["width"])
    return raw_item


def main():
    parser = argparse.ArgumentParser(description="Build no-visual slot/quality GNN data for Liu external tests.")
    parser.add_argument("--input-features", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("-confidence", "--confidence", type=float, default=0.10)
    parser.add_argument("--active-iou-threshold", type=float, default=0.50)
    parser.add_argument("--slot-iou-threshold", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    parser.add_argument("--ios-threshold", type=float, default=0.80)
    parser.add_argument("--k-arch", type=int, default=4)
    parser.add_argument("-k", "--k-neighbors", type=int, default=9)
    parser.add_argument("--y-penalty", type=float, default=3.0)
    parser.add_argument("--visual-dim", type=int, default=1024)
    parser.add_argument(
        "--use-stored-class-probs",
        action="store_true",
        help="Use stored pred_all_class_probs when present. Default forces label+score fallback for train/test consistency.",
    )
    args = parser.parse_args()

    coco_gt = COCO(args.gt_json)
    filename_to_id = build_filename_to_id_map(coco_gt)
    cat_ids = coco_gt.getCatIds(catNms=CLASS_NAMES)
    coco_id_to_model_idx = {
        cat["id"]: CLASS_NAMES.index(cat["name"])
        for cat in coco_gt.loadCats(cat_ids)
        if cat["name"] in CLASS_NAMES
    }

    gt_data = {}
    for img_id in coco_gt.getImgIds():
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            label = coco_id_to_model_idx.get(ann["category_id"])
            if label is None:
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(label)
        gt_data[img_id] = {
            "gt_bboxes_np": np.asarray(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32),
            "gt_labels_np": np.asarray(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64),
        }

    raw_list = load_raw_features(Path(args.input_features))
    graphs = []
    skipped = 0
    fallback_count = 0
    visual_missing_count = 0
    total_nodes = 0
    for raw_item in tqdm(raw_list, desc="building"):
        standardized = standardize_raw_item(raw_item, filename_to_id, coco_gt)
        if standardized is None or standardized["img_id"] not in gt_data:
            skipped += 1
            continue
        graph = build_graph_from_features(
            standardized,
            gt_data[standardized["img_id"]],
            min_score=args.confidence,
            active_iou_threshold=args.active_iou_threshold,
            slot_iou_threshold=args.slot_iou_threshold,
            iou_threshold_graph=args.iou_threshold,
            ios_threshold_graph=args.ios_threshold,
            k_arch=args.k_arch,
            k_spatial=args.k_neighbors,
            y_penalty=args.y_penalty,
            visual_dim=args.visual_dim,
            force_fallback_prior=not args.use_stored_class_probs,
        )
        if graph is None:
            skipped += 1
            continue
        fallback_count += int(bool(graph.used_fallback_prior))
        visual_missing_count += int(bool(graph.visual_missing))
        total_nodes += graph.num_nodes
        graphs.append(graph)

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)
    print("\nBuild summary")
    print(f" - input images: {len(raw_list)}")
    print(f" - graphs saved: {len(graphs)}")
    print(f" - skipped: {skipped}")
    print(f" - nodes saved: {total_nodes}")
    print(f" - fallback prior graphs: {fallback_count}/{len(graphs)}")
    print(f" - visual missing graphs: {visual_missing_count}/{len(graphs)}")
    print(f" - output: {output_path}")


if __name__ == "__main__":
    main()
