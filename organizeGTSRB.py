import os
import csv
import shutil


def organize_testset(csv_path, src_dir, target_dir):
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 读取CSV文件
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # 跳过表头

        for row in reader:
            # 解析字段（根据提供的Excel示例调整索引位置）
            filename = row[0]
            class_id = row[7]  # ClassId在第八列

            if int(class_id)<10:
                # 创建类别目录
                class_dir = os.path.join(target_dir, f'0000{class_id}')
            else:
                class_dir = os.path.join(target_dir, f'000{class_id}')
            os.makedirs(class_dir, exist_ok=True)

            # 移动文件
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(class_dir, filename)
            shutil.copy(src_path, dst_path)




import os

def remove_csv_files(root_dir):
    csvcount=0
    for class_dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    os.remove(os.path.join(dir_path, file))
                    csvcount+=1
    print(csvcount)

# 使用示例
#remove_csv_files("./data/gtsrb/train")

# # 使用示例
# organize_testset(
#     csv_path='./data/gtsrb/GT-final_test.csv',  # 测试集标注文件路径
#     src_dir='./data/gtsrb/testimages',  # 原始测试图片目录
#     target_dir='./data/gtsrb/val'  # 整理后的目录
# )