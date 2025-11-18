import torch
import argparse
import os
import pickle
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 .pt 文件 (GNN Data 列表 或 原始特征列表)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help="要合并的.pt文件列表 (例如: fold1.pt fold2.pt ...)"
    )
    parser.add_argument(
        '--output',
        required=True,
        help="合并后的输出.pt文件路径 (例如: all_folds.pt)"
    )
    args = parser.parse_args()

    print(f"开始合并 {len(args.inputs)} 个文件...")

    all_data = []

    for file_path in tqdm(args.inputs, desc="加载文件中"):
        print(f"开始合并:{file_path}")
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，跳过。")
            continue

        try:
            with open(file_path, 'rb') as file:
                data_list = pickle.load(file)
                if isinstance(data_list, list):
                    all_data.extend(data_list)
                    print(f"  - 已加载 {len(data_list)} 个条目来自 {file_path}")
                else:
                    print(f"警告: {file_path} 不是一个列表，跳过。")
        except Exception as e:
            print(f"错误: 无法加载 {file_path}。 {e}")

    print(f"\n合并完成。总条目数: {len(all_data)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(all_data, args.output)

    print(f"已保存至: {args.output}")


if __name__ == '__main__':
    main()