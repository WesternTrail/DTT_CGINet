from PIL import Image
import os
import numpy as np

def calculate_mean_and_std(folder_paths):
    images = []

    # 遍历每个文件夹路径
    for folder_path in folder_paths:
        # 检查路径是否是文件
        if os.path.isfile(folder_path) and (folder_path.endswith(".png") or folder_path.endswith(".jpg")):
            image_path = folder_path
            image = Image.open(image_path)
            image_array = np.array(image) / 255.0  # 将像素值缩放到 [0, 1]
            images.append(image_array)
        else:
            # 遍历文件夹中的所有图像文件
            for filename in os.listdir(folder_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path)
                    image_array = np.array(image) / 255.0  # 将像素值缩放到 [0, 1]
                    images.append(image_array)

    # 将图像数据转换为 NumPy 数组
    images = np.stack(images, axis=0)

    # 计算均值和标准差
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    return mean, std

# 文件夹路径，包含多个子文件夹或图像文件
parent_folder_A_path = "../data/CDD/train/A"
parent_folder_B_path = "../data/CDD/train/B"

# 获取所有子文件夹的路径或图像文件路径
subfolder_paths_A = [os.path.join(parent_folder_A_path, subfolder) for subfolder in os.listdir(parent_folder_A_path) if os.path.isdir(os.path.join(parent_folder_A_path, subfolder))]
image_paths_A = [os.path.join(parent_folder_A_path, file) for file in os.listdir(parent_folder_A_path) if os.path.isfile(os.path.join(parent_folder_A_path, file)) and (file.endswith(".png") or file.endswith(".jpg"))]

subfolder_paths_B = [os.path.join(parent_folder_B_path, subfolder) for subfolder in os.listdir(parent_folder_B_path) if os.path.isdir(os.path.join(parent_folder_B_path, subfolder))]
image_paths_B = [os.path.join(parent_folder_B_path, file) for file in os.listdir(parent_folder_B_path) if os.path.isfile(os.path.join(parent_folder_B_path, file)) and (file.endswith(".png") or file.endswith(".jpg"))]

# 计算所有子文件夹或图像文件的均值和方差
mean_combined_A, std_combined_A = calculate_mean_and_std(subfolder_paths_A + image_paths_A)
mean_combined_B, std_combined_B = calculate_mean_and_std(subfolder_paths_B + image_paths_B)

# 计算A和B总和的均值和方差
mean_combined_total = np.mean([mean_combined_A, mean_combined_B], axis=0)
std_combined_total = np.mean([std_combined_A, std_combined_B], axis=0)

# 打印结果
print("Combined Mean for A and B:", mean_combined_total)
print("Combined Std for A and B:", std_combined_total)
