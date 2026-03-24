#!/bin/bash

python build_gnn_dataset.py --input-features raw_data/raw_features_ccsw_baseline_test.pt --k-arch 4 --gt-json gt_json/test.json --output-pt gnn_data/test_ccsw_sin_arch4_edge_02.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.2
python build_gnn_dataset.py --input-features raw_data/raw_features_val.pt --k-arch 4 --gt-json gt_json/val.json --output-pt gnn_data/val_sin_arch4_edge_02.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.2
python build_gnn_dataset.py --input-features raw_data/raw_features_train.pt --k-arch 4 --gt-json gt_json/train.json --output-pt gnn_data/train_sin_arch4_edge_02.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.2

python build_gnn_dataset.py --input-features raw_data/raw_features_ccsw_baseline_test.pt --k-arch 4 --gt-json gt_json/test.json --output-pt gnn_data/test_ccsw_sin_arch4_edge_01.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.1
python build_gnn_dataset.py --input-features raw_data/raw_features_val.pt --k-arch 4 --gt-json gt_json/val.json --output-pt gnn_data/val_sin_arch4_edge_01.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.1
python build_gnn_dataset.py --input-features raw_data/raw_features_train.pt --k-arch 4 --gt-json gt_json/train.json --output-pt gnn_data/train_sin_arch4_edge_01.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.1

python build_gnn_dataset.py --input-features raw_data/raw_features_ccsw_baseline_test.pt --k-arch 4 --gt-json gt_json/test.json --output-pt gnn_data/test_ccsw_sin_arch4_edge_005.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.05
python build_gnn_dataset.py --input-features raw_data/raw_features_val.pt --k-arch 4 --gt-json gt_json/val.json --output-pt gnn_data/val_sin_arch4_edge_005.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.05
python build_gnn_dataset.py --input-features raw_data/raw_features_train.pt --k-arch 4 --gt-json gt_json/train.json --output-pt gnn_data/train_sin_arch4_edge_005.pt -k 9 --iou-threshold 0.6 --ios-threshold 0.8  --y-penalty 3.0 -confidence 0.05

