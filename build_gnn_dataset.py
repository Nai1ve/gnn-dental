from typing import Optional
import torch
import numpy as np
import pickle
import json
from pycocotools.coco import COCO
from torch_geometric.data import Data
# from torch_geometric.transforms import KNNGraph #不再需要
# from torch_geometric.utils import to_undirected #不再需要
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import os
import math
import argparse
import torchvision.ops

# --- 常量定义 ---
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
BACKGROUND = len(CLASS_NAMES)  # 48
epsilon = 1e-6
IOU_THRESHOLD = 0.5


# ==============================================================================
# 1. 核心构图算法 (V4.1: 解耦 + 各向异性)
# ==============================================================================
def build_multi_relation_graph(boxes, node_centers,
                               scores,
                               iou_threshold=0.4,
                               ios_threshold=0.8,
                               k_arch=2,
                               k_spatial=9,
                               y_penalty=3.0,
                               k_vertical=1,  # [新增参数] 纵向视野
                               x_penalty=5.0):  # [新增参数] X轴惩罚系数
    """
    构建四种解耦的边关系 (V5.1)。
    """
    N = boxes.shape[0]
    device = boxes.device

    if N < 2:
        empty_edge = torch.empty((2, 0), dtype=torch.long, device=device)
        # 返回4个空边
        return empty_edge, empty_edge, empty_edge, empty_edge

    # =========================================================
    # A. 基础矩阵计算
    # =========================================================

    # 1. IoU 矩阵
    iou_matrix = torchvision.ops.box_iou(boxes, boxes)

    # 2. IoS (Intersection over Smaller) 矩阵 - 用于解决大牙套小框
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lt = torch.max(boxes[:, None, :2], boxes[:, :2])
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    min_area = torch.min(area[:, None], area[None, :])
    ios_matrix = inter / (min_area + 1e-6)

    # =========================================================
    # B. 构建 edge_index_overlap (Type 0: 重叠/抑制边)
    # =========================================================

    # 逻辑: IoU > 阈值 OR IoS > 阈值
    is_overlapping = (iou_matrix > iou_threshold) | (ios_matrix > ios_threshold)

    # 添加方向逻辑，高Score抑制低Score
    is_score_higher = (scores.unsqueeze(1) > scores.unsqueeze(0))

    mask_overlap_directed = is_overlapping & is_score_higher

    edge_index_overlap = mask_overlap_directed.nonzero(as_tuple=False).t().contiguous()

    # 定义通用禁止掩码 (重叠的和自己的，都不能是解剖邻居)
    mask_forbidden_base = is_overlapping | torch.eye(N, dtype=torch.bool, device=device)

    # =========================================================
    # C. 构建 edge_index_arch (Type 1: 牙弓/横向边)
    # =========================================================

    # 1. 各向异性坐标变换 (拉伸Y轴，惩罚垂直距离，强制看左右)
    scaled_centers_arch = node_centers.clone()
    scaled_centers_arch[:, 1] *= y_penalty

    # 2. 计算距离矩阵
    arch_dist_matrix = torch.cdist(scaled_centers_arch, scaled_centers_arch, p=2)

    # 3. 应用约束 (禁止重叠和自循环)
    arch_dist_matrix[mask_forbidden_base] = float('inf')

    # 4. 执行 k-NN
    curr_k_arch = min(k_arch, N - 1)
    if curr_k_arch > 0:
        dists, indices = torch.topk(arch_dist_matrix, k=curr_k_arch, dim=1, largest=False)
        valid_mask = (dists != float('inf'))

        source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(N, curr_k_arch)
        valid_sources = source_nodes[valid_mask]
        valid_targets = indices[valid_mask]

        edge_index_arch = torch.stack([valid_sources, valid_targets], dim=0)
    else:
        edge_index_arch = torch.empty((2, 0), dtype=torch.long, device=device)

    # =========================================================
    # D. 构建 edge_index_vertical (Type 2: 替换牙/纵向边) [新增部分]
    # =========================================================

    # 1. 各向异性坐标变换 (拉伸X轴，惩罚水平距离，强制看上下)
    scaled_centers_vert = node_centers.clone()
    scaled_centers_vert[:, 0] *= x_penalty

    # 2. 计算距离矩阵
    vert_dist_matrix = torch.cdist(scaled_centers_vert, scaled_centers_vert, p=2)

    # 3. 应用约束
    # [关键] 禁止跨颌连接 (Anti-Opposing Constraint)
    # 我们只找"同侧牙弓"内的垂直关系 (恒牙胚 <-> 乳牙)
    y_midline = torch.median(node_centers[:, 1])
    is_upper = (node_centers[:, 1] < y_midline)
    is_lower = (node_centers[:, 1] >= y_midline)

    # 构造跨颌掩码: 如果一个在上一个在下，则禁止连接
    mask_cross_jaw = (is_upper.unsqueeze(1) & is_lower.unsqueeze(0))
    mask_cross_jaw = mask_cross_jaw | mask_cross_jaw.t()  # 双向禁止

    # 合并禁止项: 基础禁止 + 跨颌禁止
    total_forbidden_vert = mask_forbidden_base | mask_cross_jaw
    vert_dist_matrix[total_forbidden_vert] = float('inf')

    # 4. 执行 k-NN (通常 k=1, 找最近的一个替换牙)
    curr_k_vert = min(k_vertical, N - 1)
    if curr_k_vert > 0:
        dists, indices = torch.topk(vert_dist_matrix, k=curr_k_vert, dim=1, largest=False)
        valid_mask = (dists != float('inf'))

        source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(N, curr_k_vert)
        valid_sources = source_nodes[valid_mask]
        valid_targets = indices[valid_mask]

        edge_index_vertical = torch.stack([valid_sources, valid_targets], dim=0)
    else:
        edge_index_vertical = torch.empty((2, 0), dtype=torch.long, device=device)

    # =========================================================
    # E. 构建 edge_index_spatial (Type 3: 空间/兜底边)
    # =========================================================

    # 使用原始 node_centers
    spatial_dist_matrix = torch.cdist(node_centers, node_centers, p=2)
    spatial_dist_matrix.fill_diagonal_(float('inf'))

    curr_k_sp = min(k_spatial, N - 1)
    _, indices_sp = torch.topk(spatial_dist_matrix, k=curr_k_sp, dim=1, largest=False)

    source_nodes_sp = torch.arange(N, device=device).unsqueeze(1).expand(N, curr_k_sp)
    edge_index_spatial = torch.stack([source_nodes_sp.flatten(), indices_sp.flatten()], dim=0)

    # 返回 4 种边
    return edge_index_overlap, edge_index_arch, edge_index_vertical, edge_index_spatial


# ==============================================================================
# 2. 辅助工具函数
# ==============================================================================
def calculate_iou_np(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
    return iou


def generate_gnn_y_labels_local(pred_bboxes_np, pred_scores_np,
                                gt_bboxes_np, gt_labels_np):
    """为GNN节点生成y目标标签 (Ground Truth Assignment)"""
    num_preds = pred_bboxes_np.shape[0]
    num_gts = gt_bboxes_np.shape[0]
    y_vector_np = np.full(num_preds, BACKGROUND, dtype=int)

    if num_preds == 0 or num_gts == 0:
        return y_vector_np

    sort_inds = np.argsort(pred_scores_np)[::-1]
    gt_matched = np.zeros(num_gts, dtype=bool)

    # 计算 IoU 矩阵 [Num_Preds, Num_GT]
    iou_matrix = np.array([
        [calculate_iou_np(pred_bboxes_np[i], gt_box) for gt_box in gt_bboxes_np]
        for i in sort_inds
    ])

    for i_sorted in range(num_preds):
        pred_original_index = sort_inds[i_sorted]
        best_gt_idx, max_iou = -1, 0.0

        for j in range(num_gts):
            if not gt_matched[j] and iou_matrix[i_sorted, j] >= max_iou:
                max_iou = iou_matrix[i_sorted, j]
                best_gt_idx = j

        if best_gt_idx != -1 and max_iou >= IOU_THRESHOLD:
            gt_matched[best_gt_idx] = True
            gt_label = gt_labels_np[best_gt_idx]
            y_vector_np[pred_original_index] = gt_label

    return y_vector_np


# ==============================================================================
# 3. 主图构建函数 (已更新为使用 build_multi_relation_graph)
# ==============================================================================
def build_graph_from_features(raw_data: dict, gt_data: dict,
                              k_neighbors: int, min_score_threshold: int,
                              iou_threshold_graph: float, y_penalty: float,ios_threshold_graph:float) -> Optional[Data]:
    """
    从原始特征构建单个 AnatomyGAT Data对象。
    """
    # 1. 提取原始数据
    pred_bboxes_np = raw_data['pred_bboxes']
    pred_scores_np = raw_data['pred_scores']
    pred_labels_np = raw_data['pred_labels']
    x_cls_np = raw_data['x_cls']
    pred_all_class_probs = raw_data['pred_all_class_probs']

    gt_bboxes_np = gt_data['gt_bboxes_np']
    gt_labels_np = gt_data['gt_labels_np']

    img_h, img_w = raw_data['ori_shape'][0], raw_data['ori_shape'][1]
    num_preds = pred_bboxes_np.shape[0]

    if num_preds == 0:
        return None

    # 过滤低置信度
    score_mask = pred_scores_np >= min_score_threshold
    num_preds = np.sum(score_mask)

    if num_preds == 0:  # 再次检查过滤后是否为空
        return None

    # 2. 生成 GNN 的 y 标签
    y_vector_np = generate_gnn_y_labels_local(
        pred_bboxes_np, pred_scores_np, gt_bboxes_np, gt_labels_np
    )
    y = y_vector_np[score_mask]
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 3. 特征工程
    final_bboxes_tensor = torch.tensor(pred_bboxes_np[score_mask], dtype=torch.float)  # xyxy 绝对坐标
    final_scores_tensor = torch.tensor(pred_scores_np[score_mask], dtype=torch.float)
    x_cls_tensor = torch.tensor(x_cls_np[score_mask], dtype=torch.float)
    final_raw_labels_tensor = torch.tensor(pred_labels_np[score_mask], dtype=torch.int)
    final_pred_all_class_probs = torch.tensor(pred_all_class_probs[score_mask], dtype=torch.float)

    feature_list_graph = []
    feature_list_prior = []
    pos_list = []  # 绝对像素坐标 center_x, center_y

    # 计算所有框的中心点
    centers_x = (final_bboxes_tensor[:, 0] + final_bboxes_tensor[:, 2]) / 2
    centers_y = (final_bboxes_tensor[:, 1] + final_bboxes_tensor[:, 3]) / 2
    pos_abs = torch.stack([centers_x, centers_y], dim=1)  # [N, 2]

    # 计算重心
    centroid = torch.mean(pos_abs, dim=0)  # [mean_x, mean_y]

    for i in range(num_preds):
        bbox = final_bboxes_tensor[i]  # xyxy
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x_center, y_center= centers_x[i],centers_y[i]
        aspect_ratio = w / (h + 1e-6)

        # 几何特征归一化
        geom_shape_features = torch.tensor(
            [(x_center / img_w) * 2 - 1, (y_center / img_h) * 2 - 1, (w / img_w) * 2 - 1, (h / img_h) * 2 - 1])
        area_feature = torch.tensor([(w * h) / (img_w * img_h)]) * 2 - 1
        aspect_ratio_features = torch.tensor([math.log(aspect_ratio.item())])
        confidence_score = final_scores_tensor[i].unsqueeze(0) * 2 - 1
        pred_all_class_prob = final_pred_all_class_probs[i]

        # [新增特征] 重心相对坐标 (2维) -> 总共 8维
        delta_x = x_center - centroid[0]
        delta_y = y_center - centroid[1]
        rel_coord = torch.tensor([
            (delta_x / img_w) * 2,  # 归一化
            (delta_y / img_h) * 2
        ])


        # 几何特征 (8维: cx, cy, w, h, area, ratio,x重心，y重心)
        x_g = torch.cat([
            geom_shape_features,  # 4D
            area_feature,  # 1D
            aspect_ratio_features,  # 1D
            rel_coord# 2D
        ])
        feature_list_graph.append(x_g)

        # 先验特征 (50维: 49 class probs + 1 score)
        x_prior = torch.cat([
            pred_all_class_prob,
            confidence_score
        ])
        feature_list_prior.append(x_prior)

        pos_list.append(torch.tensor([x_center, y_center]))

    # 转换为 Tensor
    x_graph_features = torch.stack(feature_list_graph).float()  # [N, 6]
    x_visual_features = F.normalize(x_cls_tensor, p=2, dim=-1).float()  # [N, 1024]
    x_prior_features = torch.stack(feature_list_prior).float()  # [N, 50]

    pos = torch.stack(pos_list).float()  # [N, 2] 绝对像素坐标 (用于构图)

    # 调用 构图算法 ---
    edge_overlap, edge_arch, e_vert,edge_spatial = build_multi_relation_graph(
        boxes=final_bboxes_tensor,  # xyxy
        node_centers=pos,  # cx, cy
        scores=final_scores_tensor,
        iou_threshold=iou_threshold_graph,
        ios_threshold=ios_threshold_graph,
        k_arch=4,  # 牙弓找左右邻居
        k_spatial=k_neighbors,  # 空间找 k 个
        y_penalty=y_penalty,  # 垂直惩罚
        k_vertical=1,#替换牙数量
        x_penalty=5.0
    )

    # 归一化位置 (用于可视化或额外的几何特征，如果需要)
    pos_normalized = torch.stack([
        torch.tensor([(p[0] / img_w) * 2 - 1, (p[1] / img_h) * 2 - 1]) for p in pos
    ]).float()

    # --- 4. 组装并返回 ---
    data_final = Data(
        # 特征
        x_visual=x_visual_features,  # [N, 1024]
        x_geom=x_graph_features,  # [N, 6] (注意：这里我将其重命名为 x_geom 以匹配 model.py)
        x_prior=x_prior_features,  # [N, 50]

        # 边索引 (三种类型)
        edge_index_overlap=edge_overlap,
        edge_index_arch=edge_arch,
        edge_index_vertical=e_vert,
        edge_index_spatial=edge_spatial,

        # 标签和元数据
        y=y_tensor,
        pos=pos_normalized,
        img_id=raw_data['img_id'],  # 确保有 img_id
        img_path=raw_data['img_path'],
        ori_shape=torch.tensor(raw_data['ori_shape']),

        # 调试用元数据
        pred_bboxes_raw=final_bboxes_tensor,
        pred_scores_raw=final_scores_tensor,
        pred_labels_raw=final_raw_labels_tensor
    )
    return data_final


# ==============================================================================
# 4. 主执行函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="离线GNN数据构建脚本 (AnatomyGAT V4.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-features', required=True, help="原始特征.pt")
    parser.add_argument('--gt-json', required=True, help="GT .json")
    parser.add_argument('--output-pt', required=True, help="输出.pt")

    # GNN 构建参数
    parser.add_argument('-k', '--k-neighbors', type=int, default=9, help="空间边的 k 值 (k_spatial)")
    parser.add_argument('--iou-threshold', type=float, default=0.4, help="重叠边的 IoU 阈值")
    parser.add_argument('--ios-threshold', type=float, default=0.8, help="重叠边的 IoU 阈值")
    parser.add_argument('--y-penalty', type=float, default=3.0, help="牙弓边的垂直距离惩罚系数")
    parser.add_argument('-confidence', type=float, default=0.3, help='节点置信度过滤阈值')

    args = parser.parse_args()

    # 1. 加载原始特征
    print(f"正在加载原始特征: {args.input_features}")
    try:
        raw_data_list = torch.load(args.input_features, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}");
        return

    # 2. 加载 GT
    print(f"正在加载 GT: {args.gt_json}")
    try:
        coco_gt = COCO(args.gt_json)
    except Exception as e:
        print(f"加载失败: {e}");
        return

    # 构建映射
    coco_id_to_model_idx = {}
    cat_ids = coco_gt.getCatIds(catNms=CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    for cat in cats:
        if cat['name'] in CLASS_NAMES:
            coco_id_to_model_idx[cat['id']] = CLASS_NAMES.index(cat['name'])

    gt_data_map = {}
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_bboxes_xyxy, gt_labels_model_idx = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            model_idx = coco_id_to_model_idx.get(ann['category_id'])
            if model_idx is not None:
                gt_bboxes_xyxy.append([x, y, x + w, y + h])
                gt_labels_model_idx.append(model_idx)
        gt_data_map[img_id] = {
            'gt_bboxes_np': np.array(gt_bboxes_xyxy) if gt_bboxes_xyxy else np.empty((0, 4)),
            'gt_labels_np': np.array(gt_labels_model_idx) if gt_labels_model_idx else np.empty((0,))
        }

    # 3. 构建图
    final_gnn_list = []
    print(f"开始构建 AnatomyGAT 图 (k_sp={args.k_neighbors}, y_penalty={args.y_penalty})...")

    for raw_data in tqdm(raw_data_list, desc="构建中"):
        img_id = raw_data['img_id']
        if img_id not in gt_data_map: continue

        gt_data = gt_data_map[img_id]
        try:
            gnn_data = build_graph_from_features(
                raw_data, gt_data,
                k_neighbors=args.k_neighbors,
                min_score_threshold=args.confidence,
                iou_threshold_graph=args.iou_threshold,
                y_penalty=args.y_penalty,
                ios_threshold_graph=args.ios_threshold
            )
            if gnn_data is not None:
                final_gnn_list.append(gnn_data)
        except Exception as e:
            print(f"错误 (img_id {img_id}): {e}")
            # import traceback; traceback.print_exc()

    # 4. 保存
    print(f"完成。生成 {len(final_gnn_list)} 个图。")
    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(final_gnn_list, args.output_pt)
    print(f"保存至: {args.output_pt}")


if __name__ == '__main__':
    main()