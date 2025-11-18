import pickle
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="转化多个 .pkl 文件到pt。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help="要转化的.pkl文件列表 (例如: fold1.pt fold2.pt ...)"
    )

    args = parser.parse_args()

    print(f"开始合并 {len(args.inputs)} 个文件...")


    for file_path in args.inputs:
        print(f"开始合并:{file_path}")
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，跳过。")
            continue

        try:
            with open(file_path, 'rb') as file:
                data_list = pickle.load(file)
                if isinstance(data_list, list):
                    base, _ = os.path.splitext(file_path)
                    torch.save(data_list, base+'.pt')
                    print(f"  - 已保存 {len(data_list)} 个条目来自 {file_path}")
                else:
                    print(f"警告: {file_path} 不是一个列表，跳过。")
        except Exception as e:
            print(f"错误: 无法加载 {file_path}。 {e}")




if __name__ == '__main__':
    main()