import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
import cv2
import os

def adjust_contrast(image, level):
    """
    调整对比度。
    level: -100 到 100 的值，表示对比度调整的强度。
    """
    factor = (259 * (level + 255)) / (255 * (259 - level))
    return np.clip(factor * (image - 128) + 128, 0, 255)

def adjust_highlights(image, level):
    """
    调整高光。
    level: -100 到 100 的值，负值减少高光，正值增加高光。
    """
    mask = image > 128  # 高光部分
    image[mask] = np.clip(image[mask] + level, 0, 255)
    return image

def adjust_shadows(image, level):
    """
    调整阴影。
    level: -100 到 100 的值，负值减少阴影，正值增加阴影。
    """
    mask = image <= 128  # 阴影部分
    image[mask] = np.clip(image[mask] + level, 0, 255)
    return image

def adjust_whites(image, level):
    """
    调整白色区域。
    level: -100 到 100 的值，负值减少白色区域，正值增加白色区域。
    """
    mask = image > 200  # 假定白色区域的像素值较高
    image[mask] = np.clip(image[mask] + level, 0, 255)
    return image

def adjust_blacks(image, level):
    """
    调整黑色区域。
    level: -100 到 100 的值，负值减少黑色区域，正值增加黑色区域。
    """
    mask = image < 55  # 假定黑色区域的像素值较低
    image[mask] = np.clip(image[mask] + level, 0, 255)
    return image


# def visualize_results(avr_tensor, original_image_path,kernel_size,con,tempalte_path_list):
#     # 检查输入张量的形状
#     print("showing res")
#     if avr_tensor.dim() != 4 or avr_tensor.size(0) != 1 or avr_tensor.size(1) != 1:
#         raise ValueError("输入张量形状必须为 [1, 1, H, W]")
#
#     # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
#     tensor_np = avr_tensor.squeeze().cpu().detach().numpy()  # [H, W] 的二维数组
#
#     # 归一化张量值到 0-255 范围
#     tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) * 255
#
#     tensor_np = adjust_contrast(tensor_np,con)
#
#     # 转换为 PIL 图像
#     avr_tensor_img = Image.fromarray(tensor_np.astype(np.uint8))
#
#     # 读取原图并获取其尺寸
#     original_image = Image.open(original_image_path)
#     original_size = original_image.size
#
#     # avr_tensor_resized = 此处需要去除掉padding大小kernel_size
#     # 去除掉 padding 部分
#     kernel_size += 1
#     cropped_size = (avr_tensor_img.width - kernel_size + 1, avr_tensor_img.height - kernel_size + 1)
#     avr_tensor_cropped = avr_tensor_img.crop(
#         (kernel_size // 2, kernel_size // 2, cropped_size[0] + kernel_size // 2, cropped_size[1] + kernel_size // 2))
#
#     # 将 avr_tensor 缩放到与原图相同的尺寸
#     avr_tensor_resized = avr_tensor_cropped.resize(original_size, Image.BILINEAR)
#
#     # 转换为 NumPy 数组以便绘制
#     avr_tensor_resized_np = np.array(avr_tensor_resized)
#     original_image_np = np.array(original_image)
#
#     # 绘制原图、avr_tensor 以及叠加图
#     plt.figure(figsize=(15, 5))
#
#     # 绘制原图
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_image_np, cmap='gray' if original_image_np.ndim == 2 else None)
#     plt.title("Original Image")
#     plt.axis('off')
#
#     # 绘制 avr_tensor
#     plt.subplot(1, 3, 2)
#     plt.imshow(avr_tensor_resized_np, alpha=1)  # 使用热度图显示
#     plt.title("Average Tensor")
#     plt.axis('off')
#
#     # 绘制叠加图
#     plt.subplot(1, 3, 3)
#     plt.imshow(original_image_np, cmap='gray' if original_image_np.ndim == 2 else None)  # 显示原图
#
#     # 叠加显示热力图
#     plt.imshow(avr_tensor_resized_np, alpha=0.8)  # 使用热力图显示，alpha 控制透明度
#     plt.title("Overlay")
#     plt.axis('off')
#
#     plt.show()


def find_best_match(image_path_A, image_path_B, region_A, region_B):
    """
    找到两幅图像指定区域内匹配度最大的两个特征点坐标，并转换为相对于原始图像的坐标。

    参数:
    image_path_A (str): 图像A的路径
    image_path_B (str): 图像B的路径
    region_A (tuple): 图像A中指定的区域 (x1, y1, x2, y2)
    region_B (tuple): 图像B中指定的区域 (x1, y1, x2, y2)

    返回:
    tuple: 最佳匹配点的坐标 (ptA, ptB)，相对于原始图像的坐标
    """

    # 读取图片
    imgA = cv2.imread(image_path_A, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(image_path_B, cv2.IMREAD_GRAYSCALE)

    # 提取指定区域
    x1_A, y1_A, x2_A, y2_A = region_A
    x1_B, y1_B, x2_B, y2_B = region_B
    imgA_region = imgA[y1_A:y2_A, x1_A:x2_A]
    imgB_region = imgB[y1_B:y2_B, x1_B:x2_B]

    # 使用SIFT进行特征点检测和描述符计算
    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(imgA_region, None)
    kpB, desB = sift.detectAndCompute(imgB_region, None)

    # 检查是否成功提取到特征点
    if not kpA or not kpB:
        print("未能在指定区域内提取到特征点")
        return (0, 0), (0, 0)

    # 使用暴力匹配器进行匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desA, desB)

    # 如果没有匹配结果，返回默认值
    if not matches:
        print("没有找到匹配点")
        return (0, 0), (0, 0)

    # 按照匹配距离排序，选择距离最小的匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取最匹配的特征点
    best_match_A = kpA[matches[0].queryIdx].pt
    best_match_B = kpB[matches[0].trainIdx].pt

    # 将匹配点的坐标转换为原始图像的坐标，并转为整数
    best_match_A_original = (int(best_match_A[0] + x1_A), int(best_match_A[1] + y1_A))
    best_match_B_original = (int(best_match_B[0] + x1_B), int(best_match_B[1] + y1_B))

    # 返回最匹配点的原始图像坐标
    return best_match_A_original, best_match_B_original



# def find_best_match(image_path_A, image_path_B, region_A, region_B):
#     """
#     找到两幅图像指定区域内匹配度最大的两个特征点坐标，并转换为相对于原始图像的坐标。
#
#     参数:
#     image_path_A (str): 图像A的路径
#     image_path_B (str): 图像B的路径
#     region_A (tuple): 图像A中指定的区域 (x1, y1, x2, y2)
#     region_B (tuple): 图像B中指定的区域 (x1, y1, x2, y2)
#
#     返回:
#     tuple: 最佳匹配点的坐标 (ptA, ptB)，相对于原始图像的坐标
#     """
#
#     # 读取图片
#     imgA = cv2.imread(image_path_A, cv2.IMREAD_GRAYSCALE)
#     imgB = cv2.imread(image_path_B, cv2.IMREAD_GRAYSCALE)
#
#     # 提取指定区域
#     x1_A, y1_A, x2_A, y2_A = region_A
#     x1_B, y1_B, x2_B, y2_B = region_B
#     imgA_region = imgA[y1_A:y2_A, x1_A:x2_A]
#     imgB_region = imgB[y1_B:y2_B, x1_B:x2_B]
#
#     # 使用SIFT进行特征点检测和描述符计算
#     sift = cv2.SIFT_create()
#     kpA, desA = sift.detectAndCompute(imgA_region, None)
#     kpB, desB = sift.detectAndCompute(imgB_region, None)
#
#     # 使用暴力匹配器进行匹配
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(desA, desB)
#
#     # 按照匹配距离排序，选择距离最小的匹配点
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # 获取最匹配的特征点
#
#     best_match_A = kpA[matches[0].queryIdx].pt
#     best_match_B = kpB[matches[0].trainIdx].pt
#
#     # 将匹配点的坐标转换为原始图像的坐标，并转为整数
#     best_match_A_original = (int(best_match_A[0] + x1_A), int(best_match_A[1] + y1_A))
#     best_match_B_original = (int(best_match_B[0] + x1_B), int(best_match_B[1] + y1_B))
#
#     # 返回最匹配点的原始图像坐标
#     return best_match_A_original, best_match_B_original

def save_with_incremental_filename(base_path, base_filename, image_data):
    # 获取文件的基本路径和扩展名
    file_path = os.path.join(base_path, base_filename)
    file_name, file_extension = os.path.splitext(file_path)

    # 如果文件已存在，则添加数字后缀
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{file_name}_{counter}{file_extension}"
        counter += 1

    # 保存图像
    # plt.imsave(new_file_path, image_data)
    # print(f"图像已保存为: {new_file_path}")

# # 使用示例
# base_path = "demo_img/out"
# base_filename = "predict.jpg"
# avr_tensor_resized_np = ...  # 你的图像数据
#
# save_with_incremental_filename(base_path, base_filename, avr_tensor_resized_np)

def visualize_results(avr_tensor, original_image_path, kernel_size, con, template_path_list):
    # 检查输入张量的形状
    print("showing res")
    if avr_tensor.dim() != 4 or avr_tensor.size(0) != 1 or avr_tensor.size(1) != 1:
        raise ValueError("输入张量形状必须为 [1, 1, H, W]")

    # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
    tensor_np = avr_tensor.squeeze().cpu().detach().numpy()  # [H, W] 的二维数组

    # 归一化张量值到 0-255 范围
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) * 255

    tensor_np = adjust_contrast(tensor_np, con)

    # 转换为 PIL 图像
    avr_tensor_img = Image.fromarray(tensor_np.astype(np.uint8))

    # 读取原图并获取其尺寸
    original_image = Image.open(original_image_path)
    original_size = original_image.size

    # 去除掉 padding 部分
    kernel_size += 1
    cropped_size = (avr_tensor_img.width - kernel_size + 1, avr_tensor_img.height - kernel_size + 1)
    avr_tensor_cropped = avr_tensor_img.crop(
        (kernel_size // 2, kernel_size // 2, cropped_size[0] + kernel_size // 2, cropped_size[1] + kernel_size // 2))

    # 将 avr_tensor 缩放到与原图相同的尺寸
    avr_tensor_resized = avr_tensor_cropped.resize(original_size, Image.BILINEAR)

    # 转换为 NumPy 数组以便绘制
    avr_tensor_resized_np = np.array(avr_tensor_resized)
    original_image_np = np.array(original_image)

    # 创建绘图窗口，根据模板图片数量动态调整布局
    num_templates = len(template_path_list)
    plt.figure(figsize=(5 * (3 + num_templates), 5))

    # 绘制原图
    plt.subplot(1, 3 + num_templates, 1)
    plt.imshow(original_image_np, cmap='gray' if original_image_np.ndim == 2 else None)
    plt.title("Original Image")
    plt.axis('off')

    # 绘制 avr_tensor
    plt.subplot(1, 3 + num_templates, 2)
    plt.imshow(avr_tensor_resized_np, alpha=1)  # 使用热度图显示
    plt.title("predict")
    # plt.imsave("demo_img/out/predict.jpg",avr_tensor_resized_np)

    base_path = "demo_img/out"
    base_filename = "predict.jpg"
    # avr_tensor_resized_np = ...  # 你的图像数据
    save_with_incremental_filename(base_path, base_filename, avr_tensor_resized_np)

    plt.axis('off')

    # 绘制叠加图
    plt.subplot(1, 3 + num_templates, 3)
    plt.imshow(original_image_np, cmap='gray' if original_image_np.ndim == 2 else None)  # 显示原图
    plt.imshow(avr_tensor_resized_np, alpha=0.1)  # 使用热力图显示，alpha 控制透明度
    plt.title("Overlay")
    plt.axis('off')

    # 绘制模板图片
    for i, template_path in enumerate(template_path_list):
        template_image = Image.open(template_path)
        template_image_np = np.array(template_image)
        plt.subplot(1, 3 + num_templates, 4 + i)
        plt.imshow(template_image_np, cmap='gray' if template_image_np.ndim == 2 else None)
        plt.title(f"target")
        plt.axis('off')

    plt.show()
    return avr_tensor_resized_np


def resize_tensors(output_img, output_tem, target_size=448):
    # 获取批量大小、原始张量的高度和宽度
    bz, original_height, original_width = output_img.shape[0], output_img.shape[2], output_img.shape[3]

    # 计算两个方向的缩放因子
    height_scale = target_size / original_height
    width_scale = target_size / original_width

    # 调整 output_img 到目标大小，保留批量维度
    output_img_resized = F.interpolate(output_img, size=(target_size, target_size), mode='bilinear',
                                       align_corners=False)

    # 计算 output_tem 应该调整的新宽度和高度
    new_width = int(output_tem.shape[3] * width_scale)
    new_height = int(output_tem.shape[2] * height_scale)

    # 调整 output_tem 的宽度和高度，保留批量维度
    output_tem_resized = F.interpolate(output_tem, size=(new_height, new_width), mode='bilinear',
                                       align_corners=False)

    return output_img_resized, output_tem_resized, height_scale, width_scale

# 图像预处理函数
preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


from PIL import Image
import torch
import torchvision.transforms as transforms

def resize_tensor_to_image(template_tensor, W,H):
    """
    调整 template_tensor 的大小，使其与 template_path 指定图片的分辨率相同。

    参数:
    - template_tensor (torch.Tensor): 要调整大小的张量，形状为 (C, H, W) 或 (N, C, H, W)。
    - template_path (str): 图片的路径。

    返回:
    - torch.Tensor: 调整大小后的张量，形状为 (N, C, H', W')。
    # 示例用法
    template_tensor = torch.randn(1, 3, 224, 224)  # 假设原始张量大小为 224x224
    template_path = "path/to/your/image.jpg"  # 替换为你的图片路径

    resized_tensor = resize_tensor_to_image(template_tensor, template_path)
    print(resized_tensor.shape)  # 打印调整大小后的张量形状
    """
    # 使用 PIL 加载图片
    # image = Image.open(template_path)

    # 获取图片的宽度和高度
    image_width = W
    image_height = H

    # 如果输入的 template_tensor 是 4D (N, C, H, W)，转换为 3D (C, H, W)
    batch_dim = False
    if len(template_tensor.shape) == 4:
        batch_dim = True
        template_tensor = template_tensor.squeeze(0)  # 去掉批次维度

    # 定义调整大小的转换操作
    resize_transform = transforms.Resize((image_height, image_width))

    # 调整模板张量的大小
    # 先转换为 PIL Image，然后应用 resize，再转回张量
    template_tensor_resized = resize_transform(transforms.ToPILImage()(template_tensor)).convert("RGB")

    # 将调整大小后的 PIL 图像转换回张量
    template_tensor_resized = transforms.ToTensor()(template_tensor_resized)

    # 如果原始输入是 4D 张量，则在返回前重新添加批次维度
    if batch_dim:
        template_tensor_resized = template_tensor_resized.unsqueeze(0)

    return template_tensor_resized


import numpy as np
import matplotlib.pyplot as plt


def plot_res_list(res_list,con):
    """
    绘制res_list中的所有单通道图片，同时绘制均值图，并对每张图片进行归一化处理。

    参数:
    - res_list: 包含单通道图片的列表，已处理过的NumPy数组格式

    返回:
    - res_list: 输入的图片列表
    """

    # 定义归一化函数，将数据缩放到0-255
    def normalize_to_255(image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / (max_val - min_val) * 255
        return normalized_image

    # 对每张图片进行归一化处理
    normalized_images = [normalize_to_255(img) for img in res_list]

    # 计算归一化后的均值图
    # mean_image = np.mean(normalized_images, axis=0)

    # 创建子图，+1 是为了放置均值图
    num_images = len(normalized_images)
    fig, axes = plt.subplots(1, num_images + 1, figsize=(15, 5))

    # 绘制所有图片
    adjust_contrast_img = []
    for i, img in enumerate(normalized_images):
        img = adjust_contrast(img,con)
        adjust_contrast_img.append(img)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')

    mean_image = np.mean(adjust_contrast_img, axis=0)

    # 绘制均值图
    axes[-1].imshow(mean_image, cmap='gray', vmin=0, vmax=255)
    axes[-1].set_title('Mean Image')
    axes[-1].axis('off')

    # 调整子图间距
    plt.tight_layout()
    plt.show()

    return res_list

import torch

# 假设 density_map 是生成的密度图张量，形状为 (H, W)
def count_from_density_map(density_map):
    # 确保密度图张量位于CPU上并转换为numpy数组
    density_map = density_map.detach().cpu().numpy()
    density_map = density_map/255

    # density_map = density_map/

    # 对密度图的所有像素值求和，得到目标总数
    total_count = density_map.sum()/800000

    return total_count

# # 示例调用
# density_map = torch.randn(1, 1, 448, 448)  # 替换为实际的密度图
# total_count = count_from_density_map(density_map)
# print(f"Estimated total count: {total_count}")


def visualize_feature_maps(feature_maps, num_channels=16):
    # 确保特征图的通道数量大于等于 num_channels
    num_channels = min(feature_maps.shape[1], num_channels)

    # 将特征图从张量转换为 CPU 数据，并转换为 numpy 格式
    feature_maps = feature_maps.detach().cpu().numpy()

    # 只选取前 num_channels 个通道
    feature_maps = feature_maps[0, :num_channels, :, :]  # 选取第一张图片的前 num_channels 个通道

    # 创建 4x4 的子图窗口
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.axis('off')
        else:
            ax.remove()

    plt.tight_layout()
    plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def threshold_and_find_connected_components(gray_image, threshold_value=127):
    # 第一步：阈值分割
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # 第二步：查找联通区域
    labeled_image = label(binary_image, connectivity=2)  # 查找2连通区域
    regions = regionprops(labeled_image)

    # 第三步：创建一个彩色标记图用于可视化
    colored_labeled_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)
    for region in regions:
        # 用随机颜色填充每个连通区域
        color = np.random.randint(0, 255, size=3)
        for coord in region.coords:
            colored_labeled_image[coord[0], coord[1]] = color

    # 第四步：可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始灰度图
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 阈值分割图
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title(f'Thresholded Image (Threshold={threshold_value})')
    axes[1].axis('off')

    # 标记联通区域的彩色图
    axes[2].imshow(colored_labeled_image)
    axes[2].set_title(f'Connected Components (Count={len(regions)})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 返回连通区域的数量
    return len(regions)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import math

def otsu_threshold_and_visualize(gray_image):
    # 使用大津法确定全局阈值
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, binary_image = cv2.threshold(gray_image, 0, 255, 170)
    # _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
    print("cv2.THRESH_BINARY + cv2.THRESH_OTSU:",cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找连通区域
    labeled_image = label(binary_image, connectivity=2)
    regions = regionprops(labeled_image)

    # 生成彩色标记图
    colored_labeled_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)
    for region in regions:
        color = np.random.randint(0, 255, size=3)
        for coord in region.coords:
            colored_labeled_image[coord[0], coord[1]] = color

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title(f'Otsu Thresholding')
    axes[1].axis('off')

    axes[2].imshow(colored_labeled_image)
    axes[2].set_title(f'Connected Components (Count={len(regions)})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return len(regions)


# # 示例调用
# gray_image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)  # 使用随机图像替代实际的灰度图
# total_regions = threshold_and_find_connected_components(gray_image, threshold_value=127)
# print(f"Total connected regions: {total_regions}")


# def non_maximum_suppression(gray_image, window_size):
#     # 确保输入为 np.array
#     gray_image = np.array(gray_image)
#
#     # 获取图像的尺寸
#     H, W = gray_image.shape
#
#     # 初始化输出结果矩阵，大小与输入图像相同
#     nms_result = np.zeros_like(gray_image)
#
#     # 获取滑动窗口的大小
#     win_h, win_w = window_size
#
#     # 对图像进行滑动窗口操作
#     for y in range(0, H, win_h):
#         for x in range(0, W, win_w):
#             # 提取当前窗口区域的坐标范围
#             y1, y2 = y, min(y + win_h, H)
#             x1, x2 = x, min(x + win_w, W)
#
#             # 提取当前窗口的子图
#             window = gray_image[y1:y2, x1:x2]
#
#             # 找到窗口中的最大值及其位置
#             max_val = np.max(window)
#             max_pos = np.unravel_index(np.argmax(window), window.shape)
#
#             # 在原图的相应位置标记最大值
#             nms_result[y1:y2, x1:x2] = 0  # 先将窗口区域设置为0（抑制）
#             nms_result[y1 + max_pos[0], x1 + max_pos[1]] = max_val  # 仅保留局部最大值
#
#     return nms_result

# # 示例调用
# gray_image = np.random.randint(0, 255, size=(100, 100)).astype(np.uint8)  # 生成随机灰度图
# nms_result = non_maximum_suppression(gray_image, window_size=(5, 5))
#
# # 显示结果
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(gray_image, cmap='gray')
# ax[0].set_title('Original Gray Image')
#
# ax[1].imshow(nms_result, cmap='gray')
# ax[1].set_title('After Non-Maximum Suppression')
#
# plt.show()
def smart_non_maximum_suppression(gray_image, window_size):
    # 确保输入为 np.array
    gray_image = np.array(gray_image)

    # 获取图像的尺寸
    H, W = gray_image.shape

    # 初始化输出结果矩阵，大小与输入图像相同
    nms_result = np.zeros_like(gray_image)

    # 获取滑动窗口的大小
    win_h, win_w = window_size

    # 对图像进行滑动窗口操作
    for y in range(0, H, win_h):
        for x in range(0, W, win_w):
            # 提取当前窗口区域的坐标范围
            y1, y2 = y, min(y + win_h, H)
            x1, x2 = x, min(x + win_w, W)

            # 提取当前窗口的子图
            window = gray_image[y1:y2, x1:x2]

            # 找到窗口中的最大值及其位置
            max_val = np.max(window)
            max_pos = np.unravel_index(np.argmax(window), window.shape)

            # 检查极大值是否位于窗口的边缘
            if max_pos[0] == 0 or max_pos[0] == window.shape[0] - 1 or \
               max_pos[1] == 0 or max_pos[1] == window.shape[1] - 1:
                # 如果极大值位于窗口的边缘，则跳过当前窗口
                continue

            # 如果极大值不在边缘，则保留整个窗口的像素
            # nms_result[y1:y2, x1:x2] = window
            nms_result[y1:y2, x1:x2] = 0  # 先将窗口区域设置为0（抑制）
            nms_result[y1 + max_pos[0], x1 + max_pos[1]] = max_val  # 仅保留局部最大值

    return nms_result


def split_image_into_regions(image_path, M, N, region_size=(100, 100), edge_percentage=10):
    """
    根据图像中的M x N平铺出的中心点，返回每个区域的模板、中心点和区域坐标，
    排除图像边缘一定比例的区域。

    参数：
    image_path (str): 输入图像的路径
    M (int): 横向分割块数
    N (int): 纵向分割块数
    region_size (tuple): 每个区域的尺寸 (宽, 高)，默认 (100, 100)
    edge_percentage (int): 要排除的边缘区域百分比，默认为10%

    返回：
    list: 每个区域的模板、中心点和区域坐标
    """
    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]  # 获取图像的高度和宽度

    # 计算边缘要忽略的像素数
    edge_pixels_h = int(h * edge_percentage / 100)
    edge_pixels_w = int(w * edge_percentage / 100)

    # 计算排除边缘后的有效区域
    valid_h = h - 2 * edge_pixels_h  # 有效区域高度
    valid_w = w - 2 * edge_pixels_w  # 有效区域宽度

    # 计算每个中心点的步长
    step_h = valid_h // M
    step_w = valid_w // N

    # 计算区域的宽高
    region_width, region_height = region_size

    # 用来存储每个区域的模板、中心点和区域坐标
    regions = []

    # 双重循环遍历每个中心点
    for i in range(M):
        for j in range(N):
            # 计算每个区域的中心点
            center_x = edge_pixels_w + j * step_w + step_w // 2
            center_y = edge_pixels_h + i * step_h + step_h // 2

            # 计算区域的左上角坐标
            x1 = max(center_x - region_width // 2, edge_pixels_w)
            y1 = max(center_y - region_height // 2, edge_pixels_h)

            # 计算区域的右下角坐标
            x2 = min(center_x + region_width // 2, w - edge_pixels_w)
            y2 = min(center_y + region_height // 2, h - edge_pixels_h)

            # 提取区域的模板
            template = image[y1:y2, x1:x2]

            # 将区域信息存储到列表中
            regions.append((template, (center_x, center_y), (x1, y1, x2, y2)))

    return regions