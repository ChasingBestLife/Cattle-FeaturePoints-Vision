"""
*****************************************************
* 划分数据集：80%训练集+20%测试集
*****************************************************
"""
import os
import json
import shutil
import random

'''
--------------------------------------
- 划分数据集
--------------------------------------
'''

def split_dataset(source_folder, train_folder, val_folder, train_ratio=0.8, seed=42, image_extensions=None):
    """
    将数据集划分为训练集和验证集。

    参数:
        source_folder (str): 源数据集文件夹路径。
        train_folder (str): 训练集目标文件夹路径。
        val_folder (str): 验证集目标文件夹路径。
        train_ratio (float): 训练集所占比例，默认为0.8。
        seed (int): 随机种子，用于保证结果可重复，默认为42。
        image_extensions (list): 图像文件的扩展名列表，默认为['.png', '.jpg', '.jpeg', '.bmp', '.gif']。
    """
    # 设置随机种子
    random.seed(seed)

    # 如果未指定图像扩展名，则使用默认值
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    # 创建目标文件夹（如果不存在）
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 获取源文件夹中的所有图像文件名
    image_files = [f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    # 如果没有找到图像文件，抛出异常
    if not image_files:
        raise ValueError(f"在源文件夹 '{source_folder}' 中没有找到任何图像文件。")

    # 打乱文件列表以随机划分
    random.shuffle(image_files)

    # 计算训练集和验证集的大小
    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    val_size = total_images - train_size

    # 划分训练集和验证集
    train_images = image_files[:train_size]
    val_images = image_files[train_size:]

    # 将图像复制到目标文件夹
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

    for img in val_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

    print(f"划分完成！训练集有 {train_size} 张图像，验证集有 {val_size} 张图像。")
    return train_size, val_size



def copy_to_labes(source_labels_folder, source_images_folder, target_labels_folder):
    # 确保目标文件夹存在
    os.makedirs(target_labels_folder, exist_ok=True)

    # 遍历images/train文件夹中的所有文件
    for image_file in os.listdir(source_images_folder):
        # 获取文件名（不包含扩展名）
        image_name = os.path.splitext(image_file)[0]

        # 构造labels_pre文件夹中对应的标签文件路径
        label_file = os.path.join(source_labels_folder, image_name + ".txt")

        # 检查标签文件是否存在
        if os.path.exists(label_file):
            # 构造目标标签文件路径
            target_label_file = os.path.join(target_labels_folder, image_name + ".txt")

            # 复制标签文件到目标文件夹
            shutil.copy(label_file, target_label_file)
            print(f"Copied {label_file} to {target_label_file}")
        else:
            print(f"No matching label file found for {image_file}")

# 示例用法
if __name__ == "__main__":
    images_source_folder = r'dataset/cow_final/images'
    labels_source_folder = r"dataset/cow_final/labels"

    images_folder_train = r"dataset/FivePoints_datasets/images/train"
    labels_folder_train = r"dataset/FivePoints_datasets/labels/train"
    images_folder_val = r"dataset/FivePoints_datasets/images/val"
    labels_folder_val = r"dataset/FivePoints_datasets/labels/val"

    # images划分
    split_dataset(images_source_folder, images_folder_train, images_folder_val)

    # # 根据images划分情况划分labels
    copy_to_labes(labels_source_folder, images_folder_train, labels_folder_train)
    copy_to_labes(labels_source_folder, images_folder_val, labels_folder_val)








