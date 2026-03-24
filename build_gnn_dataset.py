from typing import Optional
import torch
import numpy as np
from pycocotools.coco import COCO
from torch_geometric.data import Data
import torch.nn.functional as F
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


def fourier_encode(x, num_bands=4):
    """
    x: [N] 或 [N, 1] 的坐标张量
    num_bands: 频率带数量
    返回: [N, num_bands * 2]
    """
    x = x.unsqueeze(-1)  # [N, 1]
    device = x.device
    freqs = torch.pow(2, torch.arange(num_bands, dtype=torch.float, device=device))
    x_freq = x * freqs * torch.pi
    return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)


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
                               k_vertical=1,
                               x_penalty=5.0):
    """
    构建四种解耦的边关系 (V5.1)。
    """
    N = boxes.shape[0]
    device = boxes.device

    if N < 2:
        empty_edge = torch.empty((2, 0), dtype=torch.long, device=device)
        return empty_edge, empty_edge, empty_edge, empty_edge

    # 1. IoU & IoS 矩阵
    iou_matrix = torchvision.ops.box_iou(boxes, boxes)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lt = torch.max(boxes[:, None, :2], boxes[:, :2])
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    min_area = torch.min(area[:, None], area[None, :])
    ios_matrix = inter / (min_area + 1e-6)

    # 2. edge_index_overlap
    is_overlapping = (iou_matrix > iou_threshold) | (ios_matrix > ios_threshold)
    is_overlapping.fill_diagonal_(False)

    # [核心修改]: 彻底废弃 is_score_higher = (scores.unsqueeze(1) > scores.unsqueeze(0))
    # 允许低分候选框通过图推理“反杀”高分的错误框/伪影框。
    # 构建双向重叠边，让下游的 RGAT 根据视觉特征和几何特征(边属性)动态计算抑制权重。
    edge_index_overlap = is_overlapping.nonzero(as_tuple=False).t().contiguous()

    # mask_forbidden_base 依然需要保留，用于防止在牙弓/垂直关系中建立重叠节点的连线
    mask_forbidden_base = is_overlapping | torch.eye(N, dtype=torch.bool, device=device)

    # 3. edge_index_arch
    scaled_centers_arch = node_centers.clone()
    scaled_centers_arch[:, 1] *= y_penalty
    arch_dist_matrix = torch.cdist(scaled_centers_arch, scaled_centers_arch, p=2)
    arch_dist_matrix[mask_forbidden_base] = float('inf')

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

    # 4. edge_index_vertical
    scaled_centers_vert = node_centers.clone()
    scaled_centers_vert[:, 0] *= x_penalty
    vert_dist_matrix = torch.cdist(scaled_centers_vert, scaled_centers_vert, p=2)

    y_midline = torch.median(node_centers[:, 1])
    is_upper = (node_centers[:, 1] < y_midline)
    is_lower = (node_centers[:, 1] >= y_midline)
    mask_cross_jaw = (is_upper.unsqueeze(1) & is_lower.unsqueeze(0))
    mask_cross_jaw = mask_cross_jaw | mask_cross_jaw.t()  # 双向禁止

    total_forbidden_vert = mask_forbidden_base | mask_cross_jaw
    vert_dist_matrix[total_forbidden_vert] = float('inf')

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

    # 5. edge_index_spatial
    spatial_dist_matrix = torch.cdist(node_centers, node_centers, p=2)
    spatial_dist_matrix.fill_diagonal_(float('inf'))
    curr_k_sp = min(k_spatial, N - 1)
    _, indices_sp = torch.topk(spatial_dist_matrix, k=curr_k_sp, dim=1, largest=False)
    source_nodes_sp = torch.arange(N, device=device).unsqueeze(1).expand(N, curr_k_sp)
    edge_index_spatial = torch.stack([source_nodes_sp.flatten(), indices_sp.flatten()], dim=0)

    return edge_index_overlap, edge_index_arch, edge_index_vertical, edge_index_spatial,iou_matrix


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
# 3. 数据标准化与补全 (新增核心模块)
# ==============================================================================
def standardize_input_data(raw_data_item: dict,
                           filename_to_id: dict,
                           coco_gt: COCO,
                           visual_dim: int = 1024) -> Optional[dict]:
    """
    统一 YOLO 和 Faster R-CNN 的数据格式。
    1. 统一 ID: 将 filename 转换为 img_id
    2. 补全特征: 如果缺少 visual feature，生成零向量
    3. 补全 Info: 如果缺少 ori_shape，从 COCO 读取
    """

    # --- 1. ID 统一 ---
    img_id = raw_data_item.get('img_id')

    # 如果没有 img_id 或者 img_id 为空，尝试通过 filename (YOLO 常见情况)
    if img_id is None:
        raw_filename = raw_data_item.get('file_name')  # YOLO 通常有 file_name 或 img_path
        if raw_filename is None:
            # 尝试 raw_data_item['img_path']
            raw_filename = raw_data_item.get('img_path')

        if raw_filename:
            # 提取纯文件名 (e.g., 'data/imgs/123.jpg' -> '123.jpg')
            base_name = os.path.basename(raw_filename)
            if base_name in filename_to_id:
                img_id = filename_to_id[base_name]
            else:
                # 无法匹配到 GT 中的文件名，跳过
                return None
        else:
            return None  # 既无ID也无文件名，无法处理

    # 此时我们有了 img_id，从 COCO 获取权威信息
    img_info = coco_gt.loadImgs(img_id)[0]

    # --- 2. 补全原始形状 (YOLO 有时缺失) ---
    if 'ori_shape' not in raw_data_item:
        raw_data_item['ori_shape'] = (img_info['height'], img_info['width'])

    # --- 3. 补全视觉特征 (YOLO 缺失) ---
    # 检查是否有 x_cls
    num_boxes = len(raw_data_item['pred_bboxes'])
    if 'x_cls' not in raw_data_item or raw_data_item['x_cls'] is None:
        # 补全全0特征
        # 注意：后续会有 F.normalize，全0向量归一化后仍为0，是安全的
        raw_data_item['x_cls'] = np.zeros((num_boxes, visual_dim), dtype=np.float32)

    # 确保是 Numpy 数组
    if not isinstance(raw_data_item['x_cls'], np.ndarray):
        raw_data_item['x_cls'] = np.array(raw_data_item['x_cls'])

    # --- 4. 确保 img_id 写入字典 ---
    raw_data_item['img_id'] = img_id
    raw_data_item['img_path'] = img_info['file_name']  # 统一用 COCO 的路径

    return raw_data_item


# ==============================================================================
# 4. 主图构建函数
# ==============================================================================
def build_graph_from_features(raw_data: dict, gt_data: dict,
                              k_neighbors: int, min_score_threshold: float,
                              iou_threshold_graph: float, y_penalty: float,
                              ios_threshold_graph: float, k_arch: int) -> Optional[Data]:
    """
    从标准化后的特征构建 AnatomyGAT Data对象。
    """
    pred_bboxes_np = raw_data['pred_bboxes']
    pred_scores_np = raw_data['pred_scores']
    pred_labels_np = raw_data['pred_labels']
    x_cls_np = raw_data['x_cls']  # 此时必然存在
    pred_all_class_probs = raw_data['pred_all_class_probs']

    gt_bboxes_np = gt_data['gt_bboxes_np']
    gt_labels_np = gt_data['gt_labels_np']

    img_h, img_w = raw_data['ori_shape'][0], raw_data['ori_shape'][1]
    num_preds = pred_bboxes_np.shape[0]

    if num_preds == 0: return None

    # 过滤低置信度
    score_mask = pred_scores_np >= min_score_threshold
    num_preds = np.sum(score_mask)

    if num_preds == 0: return None

    # 生成 GNN 的 y 标签
    y_vector_np = generate_gnn_y_labels_local(
        pred_bboxes_np, pred_scores_np, gt_bboxes_np, gt_labels_np
    )
    y = y_vector_np[score_mask]
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 特征工程
    final_bboxes_tensor = torch.tensor(pred_bboxes_np[score_mask], dtype=torch.float)
    final_scores_tensor = torch.tensor(pred_scores_np[score_mask], dtype=torch.float)
    x_cls_tensor = torch.tensor(x_cls_np[score_mask], dtype=torch.float)
    final_raw_labels_tensor = torch.tensor(pred_labels_np[score_mask], dtype=torch.int)
    final_pred_all_class_probs = torch.tensor(pred_all_class_probs[score_mask], dtype=torch.float)

    feature_list_graph = []
    feature_list_prior = []
    pos_list = []

    centers_x = (final_bboxes_tensor[:, 0] + final_bboxes_tensor[:, 2]) / 2
    centers_y = (final_bboxes_tensor[:, 1] + final_bboxes_tensor[:, 3]) / 2
    pos_abs = torch.stack([centers_x, centers_y], dim=1)

    centroid = torch.mean(pos_abs, dim=0)
    norm_dx = (centers_x - centroid[0]) / img_w
    norm_dy = (centers_y - centroid[1]) / img_h

    enc_dx = fourier_encode(norm_dx, num_bands=4)
    enc_dy = fourier_encode(norm_dy, num_bands=4)
    rel_coord_all = torch.cat([enc_dx, enc_dy], dim=1)

    for i in range(num_preds):
        bbox = final_bboxes_tensor[i]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x_center, y_center = centers_x[i], centers_y[i]
        aspect_ratio = w / (h + 1e-6)

        geom_shape_features = torch.tensor(
            [(x_center / img_w) * 2 - 1, (y_center / img_h) * 2 - 1, (w / img_w) * 2 - 1, (h / img_h) * 2 - 1])
        area_feature = torch.tensor([(w * h) / (img_w * img_h)]) * 2 - 1
        aspect_ratio_features = torch.tensor([math.log(aspect_ratio.item())])
        confidence_score = final_scores_tensor[i].unsqueeze(0) * 2 - 1
        pred_all_class_prob = final_pred_all_class_probs[i]

        base_geom = torch.cat([geom_shape_features, area_feature, aspect_ratio_features])
        x_g = torch.cat([base_geom, rel_coord_all[i]])

        feature_list_graph.append(x_g)
        x_prior = torch.cat([pred_all_class_prob, confidence_score])
        feature_list_prior.append(x_prior)
        pos_list.append(torch.tensor([x_center, y_center]))

    x_graph_features = torch.stack(feature_list_graph).float()

    # [关键] 视觉特征归一化 (如果是填充的0向量，结果仍为0)
    x_visual_features = F.normalize(x_cls_tensor, p=2, dim=-1).float()

    x_prior_features = torch.stack(feature_list_prior).float()
    pos = torch.stack(pos_list).float()

    edge_overlap, edge_arch, e_vert, edge_spatial, iou_matrix = build_multi_relation_graph(
        boxes=final_bboxes_tensor,
        node_centers=pos,
        scores=final_scores_tensor,
        iou_threshold=iou_threshold_graph,
        ios_threshold=ios_threshold_graph,
        k_arch=k_arch,
        k_spatial=k_neighbors,
        y_penalty=y_penalty,
        k_vertical=1,
        x_penalty=5.0
    )

    edge_attr_overlap = compute_edge_attributes(edge_overlap, final_bboxes_tensor, pos, img_w, img_h, iou_matrix)
    edge_attr_arch = compute_edge_attributes(edge_arch, final_bboxes_tensor, pos, img_w, img_h)
    edge_attr_vert = compute_edge_attributes(e_vert, final_bboxes_tensor, pos, img_w, img_h)
    edge_attr_spatial = compute_edge_attributes(edge_spatial, final_bboxes_tensor, pos, img_w, img_h)

    pos_normalized = torch.stack([
        torch.tensor([(p[0] / img_w) * 2 - 1, (p[1] / img_h) * 2 - 1]) for p in pos
    ]).float()

    data_final = Data(
        x_visual=x_visual_features,
        x_geom=x_graph_features,
        x_prior=x_prior_features,
        # 记录拓扑连线 (Prior Mask)
        edge_index_overlap=edge_overlap,
        edge_index_arch=edge_arch,
        edge_index_vertical=e_vert,
        edge_index_spatial=edge_spatial,
        # 记录拓扑关系特征 (Dynamic Edge Weights 的原材料)
        edge_attr_overlap=edge_attr_overlap,
        edge_attr_arch=edge_attr_arch,
        edge_attr_vertical=edge_attr_vert,
        edge_attr_spatial=edge_attr_spatial,

        y=y_tensor,
        pos=pos_normalized,
        img_id=raw_data['img_id'],
        img_path=raw_data['img_path'],
        ori_shape=torch.tensor(raw_data['ori_shape']),
        pred_bboxes_raw=final_bboxes_tensor,
        pred_scores_raw=final_scores_tensor,
        pred_labels_raw=final_raw_labels_tensor
    )
    return data_final


def compute_edge_attributes(edge_index, boxes, centers, img_w, img_h, iou_matrix=None):
    """
    [新增核心] 将启发式连线转化为可学习的关系特征 (Relational Prior Features)
    为每条边计算: [归一化 dx, 归一化 dy, 欧氏距离, log宽度比, log高度比, (可选) IoU]
    """
    device = boxes.device
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        # 如果没有IoU矩阵传入，默认5维特征；否则6维
        dim = 5 if iou_matrix is None else 6
        return torch.empty((0, dim), dtype=torch.float, device=device)

    src, dst = edge_index[0], edge_index[1]

    # 1. 相对位置 (用图像宽高归一化)
    dx = (centers[dst, 0] - centers[src, 0]) / img_w
    dy = (centers[dst, 1] - centers[src, 1]) / img_h
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    # 2. 相对尺度 (衡量候选框大小变化)
    w_src = boxes[src, 2] - boxes[src, 0]
    h_src = boxes[src, 3] - boxes[src, 1]
    w_dst = boxes[dst, 2] - boxes[dst, 0]
    h_dst = boxes[dst, 3] - boxes[dst, 1]

    log_scale_w = torch.log((w_dst + 1e-6) / (w_src + 1e-6))
    log_scale_h = torch.log((h_dst + 1e-6) / (h_src + 1e-6))

    # 3. 基础边缘特征组合
    edge_attr = torch.stack([dx, dy, dist, log_scale_w, log_scale_h], dim=1)

    # 4. 如果是 Overlap 竞争边，额外拼接 IoU 作为先验强度
    if iou_matrix is not None:
        iou_vals = iou_matrix[src, dst].unsqueeze(1)
        edge_attr = torch.cat([edge_attr, iou_vals], dim=1)

    return edge_attr


# ==============================================================================
# 5. 主执行函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="离线GNN数据构建脚本 (兼容 YOLO/Faster R-CNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-features', required=True, help="原始特征.pt (List[Dict])")
    parser.add_argument('--gt-json', required=True, help="GT .json")
    parser.add_argument('--output-pt', required=True, help="输出.pt")

    parser.add_argument('-k', '--k-neighbors', type=int, default=9, help="空间边的 k 值")
    parser.add_argument('--iou-threshold', type=float, default=0.4, help="重叠边的 IoU 阈值")
    parser.add_argument('--ios-threshold', type=float, default=0.8, help="IoS 阈值")
    parser.add_argument('--y-penalty', type=float, default=3.0, help="牙弓边垂直距离惩罚")
    parser.add_argument('--k-arch', type=int, default=4, help="牙弓边 k 值")
    parser.add_argument('-confidence', type=float, default=0.3, help='节点置信度过滤阈值')

    # [新增] 视觉特征维度参数
    parser.add_argument('--visual-dim', type=int, default=1024, help='视觉特征维度 (若缺失则填充此维度的0向量)')

    args = parser.parse_args()

    # 1. 加载 GT (先加载GT以建立映射)
    print(f"正在加载 GT: {args.gt_json}")
    try:
        coco_gt = COCO(args.gt_json)
    except Exception as e:
        print(f"GT 加载失败: {e}")
        return

    # [新增] 建立 Filename -> ImageID 的映射 (处理 YOLO 文件名索引问题)
    print("正在建立 Filename -> ID 映射...")
    filename_to_id = {}
    for img_info in coco_gt.dataset['images']:
        # 使用 basename 以防止路径差异 (e.g. data/1.jpg vs 1.jpg)
        base_name = os.path.basename(img_info['file_name'])
        filename_to_id[base_name] = img_info['id']
    print(f"映射建立完成，共 {len(filename_to_id)} 张图片。")

    # 构建类别映射
    coco_id_to_model_idx = {}
    cat_ids = coco_gt.getCatIds(catNms=CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    for cat in cats:
        if cat['name'] in CLASS_NAMES:
            coco_id_to_model_idx[cat['id']] = CLASS_NAMES.index(cat['name'])

    # 缓存 GT 数据
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

    # 2. 加载原始特征
    print(f"正在加载输入特征: {args.input_features}")
    try:
        raw_data_list = torch.load(args.input_features)
    except Exception as e:
        print(f"输入加载失败: {e}")
        return

    # 3. 构建图
    final_gnn_list = []
    print(f"开始构建 AnatomyGAT 图...")

    # 计数统计
    missing_id_count = 0
    missing_xcls_count = 0

    for raw_data_item in tqdm(raw_data_list, desc="构建中"):

        # [核心修改] 标准化输入数据 (统一 ID, 补全特征)
        is_missing_xcls = ('x_cls' not in raw_data_item or raw_data_item['x_cls'] is None)
        standardized_data = standardize_input_data(
            raw_data_item,
            filename_to_id,
            coco_gt,
            visual_dim=args.visual_dim
        )

        if standardized_data is None:
            missing_id_count += 1
            continue

        if is_missing_xcls:
            missing_xcls_count += 1

        img_id = standardized_data['img_id']

        # 确保 GT 存在
        if img_id not in gt_data_map:
            continue
        gt_data = gt_data_map[img_id]

        try:
            gnn_data = build_graph_from_features(
                standardized_data, gt_data,
                k_neighbors=args.k_neighbors,
                min_score_threshold=args.confidence,
                iou_threshold_graph=args.iou_threshold,
                y_penalty=args.y_penalty,
                ios_threshold_graph=args.ios_threshold,
                k_arch=args.k_arch
            )
            if gnn_data is not None:
                final_gnn_list.append(gnn_data)
        except Exception as e:
            print(f"错误 (img_id {img_id}): {e}")
            import traceback;
            traceback.print_exc()

    print(f"\n构建摘要:")
    print(f" - 输入总数: {len(raw_data_list)}")
    print(f" - 成功构建: {len(final_gnn_list)}")
    print(f" - 因 ID/Filename 无法匹配跳过: {missing_id_count}")
    print(f" - 检测到缺失视觉特征 (已补全0向量): {missing_xcls_count}")

    # 4. 保存
    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(final_gnn_list, args.output_pt)
    print(f"结果已保存至: {args.output_pt}")


if __name__ == '__main__':
    main()