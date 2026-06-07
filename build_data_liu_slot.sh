#!/bin/bash

set -e

python build_gnn_dataset_liu_external.py \
  --input-features raw_data/raw_features_train.pt \
  --gt-json gt_json/train.json \
  --output-pt gnn_data/train_sin_arch4_edge_01_slot_struct_order.pt \
  -confidence 0.1 \
  --iou-threshold 0.6 \
  --ios-threshold 0.8 \
  --k-arch 4 \
  -k 9 \
  --y-penalty 3.0

python build_gnn_dataset_liu_external.py \
  --input-features raw_data/raw_features_val.pt \
  --gt-json gt_json/val.json \
  --output-pt gnn_data/val_sin_arch4_edge_01_slot_struct_order.pt \
  -confidence 0.1 \
  --iou-threshold 0.6 \
  --ios-threshold 0.8 \
  --k-arch 4 \
  -k 9 \
  --y-penalty 3.0

python build_gnn_dataset_liu_external.py \
  --input-features raw_data_liu/raw_features_ccsw_liu_test.pkl \
  --gt-json gt_json_liu/test.json \
  --output-pt gnn_data/test_ccsw_liu_test_slot_struct_order.pt \
  -confidence 0.1 \
  --iou-threshold 0.6 \
  --ios-threshold 0.8 \
  --k-arch 4 \
  -k 9 \
  --y-penalty 3.0

python build_gnn_dataset_liu_external.py \
  --input-features raw_data_liu/raw_features_ccsw_liu_test_exp.pkl \
  --gt-json gt_json_liu/test_exp.json \
  --output-pt gnn_data/test_ccsw_liu_exp_slot_struct_order.pt \
  -confidence 0.1 \
  --iou-threshold 0.6 \
  --ios-threshold 0.8 \
  --k-arch 4 \
  -k 9 \
  --y-penalty 3.0

python build_gnn_dataset_liu_external.py \
  --input-features raw_data_liu/raw_features_ccsw_liu_test_exp_open.pkl \
  --gt-json gt_json_liu/test_exp_open.json \
  --output-pt gnn_data/test_ccsw_liu_exp_open_slot_struct_order.pt \
  -confidence 0.1 \
  --iou-threshold 0.6 \
  --ios-threshold 0.8 \
  --k-arch 4 \
  -k 9 \
  --y-penalty 3.0
