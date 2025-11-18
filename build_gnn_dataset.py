from typing import Optional

import torch
import numpy as np
import pickle
import json
from pycocotools.coco import COCO
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import os
import math
import argparse


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
    """
    为GNN节点生成y目标标签
    """
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


# --- 3. 主图构建函数 ---
def build_graph_from_features(raw_data: dict, gt_data: dict, k_neighbors: int,min_score_threshold:int) -> Optional[Data]:
    """
    从原始特征构建单个GNN Data对象。
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

    score_mask = pred_scores_np >= min_score_threshold
    num_preds = np.sum(score_mask)

    # 2. 生成 GNN 的 y 标签
    y_vector_np = generate_gnn_y_labels_local(
        pred_bboxes_np, pred_scores_np, gt_bboxes_np, gt_labels_np
    )
    y = y_vector_np[score_mask]
    y_tensor = torch.tensor(y,dtype=torch.int)
    #  3. 特征工程

    # 转换所有为 Tensor
    final_bboxes_tensor = torch.tensor(pred_bboxes_np[score_mask],dtype=torch.float)
    final_scores_tensor = torch.tensor(pred_scores_np[score_mask],dtype=torch.float)
    x_cls_tensor = torch.tensor(x_cls_np[score_mask],dtype=torch.float)
    final_raw_labels_tensor = torch.tensor(pred_labels_np[score_mask],dtype=torch.int)
    final_pred_all_class_probs = torch.tensor(pred_all_class_probs[score_mask],dtype=torch.float)

    feature_list_graph = []
    feature_list_prior = []
    pos_list = []
    diag_length = math.sqrt(img_w ** 2 + img_h ** 2) + epsilon

    for i in range(num_preds):
        bbox = final_bboxes_tensor[i]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x_center = bbox[0] + w / 2
        y_center = bbox[1] + h / 2
        aspect_ratio = w / (h + 1e-6)

        geom_shape_features = torch.tensor(
            [(x_center / img_w) * 2 - 1, (y_center / img_h) * 2 - 1, (w / img_w) * 2 - 1, (h / img_h) * 2 - 1])
        area_feature = torch.tensor([(w * h) / (img_w * img_h)]) * 2 - 1
        aspect_ratio_features = torch.tensor([math.log(aspect_ratio.item())])
        confidence_score = final_scores_tensor[i].unsqueeze(0) * 2 - 1
        pred_all_class_prob = final_pred_all_class_probs[i]


        x_g = torch.cat([
            geom_shape_features,  # 4D
            area_feature,  # 1D
            aspect_ratio_features,  # 1D
            #confidence_score  # 1D
        ])  # 总共 7 维
        feature_list_graph.append(x_g)

        x_prior = torch.cat([
            pred_all_class_prob,
            confidence_score
        ])

        feature_list_prior.append(x_prior)

        pos_list.append(torch.tensor([x_center, y_center]))



    x_graph_features = torch.stack(feature_list_graph).float()  # [N, 7]
    x_visual_features = F.normalize(x_cls_tensor, p=2, dim=-1).float()  # [N, 1024]
    x_prior_features = torch.stack(feature_list_prior).float()

    pos = torch.stack(pos_list)  # [N, 2] 像素坐标
    pos_normalized = torch.stack([
        torch.tensor([(p[0] / img_w) * 2 - 1, (p[1] / img_h) * 2 - 1]) for p in pos
    ]).float()


    temp_data = Data(pos=pos)
    knn_transform = KNNGraph(k=k_neighbors)
    graph_data = knn_transform(temp_data)
    edge_index = to_undirected(graph_data.edge_index)

    edge_attr_list = []
    row, col = edge_index
    for i in range(len(row)):
        src_node_idx, dest_node_idx = row[i].item(), col[i].item()
        vec = pos[dest_node_idx] - pos[src_node_idx]
        dist = torch.norm(vec, p=2)

        normalized_dist = (dist / diag_length) * 2 - 1
        vec_norm = dist + epsilon
        norm_vec_x = vec[0] / vec_norm
        norm_vec_y = vec[1] / vec_norm

        edge_attr_list.append(torch.tensor([normalized_dist, norm_vec_x, norm_vec_y]))

    edge_attr = torch.stack(edge_attr_list).float() if edge_attr_list else torch.empty((0, 3)).float()  # [E, 3]

    # --- 4. 组装并返回 ---
    data_final = Data(
        x_graph=x_graph_features,  # [N, 7]
        x_visual=x_visual_features,  # [N, 1024]
        x_prior = x_prior_features, #[N,50]
        edge_index=edge_index,
        edge_attr=edge_attr,  # [E, 3]
        pos=pos_normalized,
        y=y_tensor,
        img_id=raw_data['img_id'],
        img_path=raw_data['img_path'],
        ori_shape=torch.tensor(raw_data['ori_shape']),
        pred_bboxes_raw=final_bboxes_tensor,
        pred_scores_raw=final_scores_tensor,
        pred_labels_raw=final_raw_labels_tensor
    )
    return data_final


# --- 4. 主执行函数 ---
def main():
    parser = argparse.ArgumentParser(
        description="离线GNN数据构建脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-features', required=True, help="从MMDetection提取的原始特征.pt文件路径")
    parser.add_argument('--gt-json', required=True, help="对应的COCO GT .json文件路径")
    parser.add_argument('--output-pt', required=True, help="GNN Data列表的输出.pt文件路径")
    parser.add_argument('-k', '--k-neighbors', type=int, default=6, help="k-NN的k值")
    parser.add_argument('-confidence', type=float,default=0.3,help='置信度阈值')
    args = parser.parse_args()

    # 1. 加载原始特征
    print(f"正在加载原始特征文件: {args.input_features}")
    try:
        raw_data_list = torch.load(args.input_features, map_location='cpu')
    except Exception as e:
        print(f"错误: 无法加载 {args.input_features}。 {e}");
        return

    # 2. 加载并索引GT
    print(f"正在加载并索引GT文件: {args.gt_json}")
    try:
        coco_gt = COCO(args.gt_json)
    except Exception as e:
        print(f"错误: 无法加载 {args.gt_json}。 {e}");
        return

    # [关键] 构建 COCO ID -> 模型索引 (0-51) 的映射
    coco_id_to_model_idx = {}
    cat_ids = coco_gt.getCatIds(catNms=CLASS_NAMES)
    cats = coco_gt.loadCats(cat_ids)
    for cat in cats:
        if cat['name'] in CLASS_NAMES:
            coco_id_to_model_idx[cat['id']] = CLASS_NAMES.index(cat['name'])

    # 将GT数据索引到字典中，以便快速查找
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
    print("GT索引构建完成。")

    # 3. 循环构建图
    final_gnn_list = []
    print(f"开始构建GNN图 (k={args.k_neighbors})...")
    for raw_data in tqdm(raw_data_list, desc="构建图中"):
        img_id = raw_data['img_id']
        if img_id not in gt_data_map:
            print(f"警告: 找不到 img_id {img_id} 的 GT 标注，跳过。")
            continue

        gt_data = gt_data_map[img_id]

        try:
            gnn_data = build_graph_from_features(raw_data, gt_data, args.k_neighbors,args.confidence)
            if gnn_data is not None:
                final_gnn_list.append(gnn_data)
        except Exception as e:
            print(f"\n错误: 处理 img_id {img_id} (路径: {raw_data['img_path']}) 时发生错误: {e}")
            # 可以在这里打印更详细的堆栈跟踪
            import traceback; traceback.print_exc()

    # 4. 保存
    print(f"\n图构建完成。总共生成 {len(final_gnn_list)} 个GNN图。")
    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(final_gnn_list, args.output_pt)
    print(f"GNN数据已成功保存至: {args.output_pt}")


if __name__ == '__main__':
    main()