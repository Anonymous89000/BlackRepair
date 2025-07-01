import os
import shutil


def move_images_to_target_class(rootdir, targetclassdir):
    # 获取一级目录下的所有子目录
    print(os.listdir(rootdir))

    for subdir in os.listdir(rootdir):
        subdir_path = os.path.join(rootdir, subdir)

        # 只处理子目录，排除targetclassdir
        if os.path.isdir(subdir_path) and subdir_path != targetclassdir:
            print(f"Processing directory: {subdir}")

            # 遍历当前子目录中的所有文件
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # 检查文件是否是图片文件，可以根据文件扩展名进一步限制
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # 移动图片文件到targetclassdir
                    target_path = os.path.join(targetclassdir, filename)
                    try:
                        # 如果文件已存在，抛出异常并停止处理
                        if os.path.exists(target_path):
                            print(filename,targetclassdir)
                            raise FileExistsError(
                                f"File {filename} already exists in {targetclassdir}, stopping process.")

                        shutil.move(file_path, target_path)
                        print(f"Moved {filename} to {targetclassdir}")
                    except FileExistsError as e:
                        print(f"Error: {e}")
                        return  # 结束脚本执行


# 示例调用：
# rootdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_BadNets'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_BadNets/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_WaNet'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_WaNet/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_Blended'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_IMAGENET10_stdvgg16_class10_Blended/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_CIFAR10_innervgg13_BadNets'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_CIFAR10_innervgg13_BadNets/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_CIFAR10_innervgg13_WaNet'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_CIFAR10_innervgg13_WaNet/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_CIFAR10_innervgg13_Blended'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_CIFAR10_innervgg13_Blended/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_GTSRB_resnet18_class43_BadNets'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_GTSRB_resnet18_class43_BadNets/true_class_0'  # 替换为目标目录名称

# rootdir = 'data/poisonedval_GTSRB_resnet18_class43_WaNet'  # 替换为你的一级目录路径
# targetclassdir = 'data/poisonedval_GTSRB_resnet18_class43_WaNet/true_class_0'  # 替换为目标目录名称

rootdir = 'data/poisonedval_GTSRB_resnet18_class43_Blended'  # 替换为你的一级目录路径
targetclassdir = 'data/poisonedval_GTSRB_resnet18_class43_Blended/true_class_0'  # 替换为目标目录名称

move_images_to_target_class(rootdir, targetclassdir)
