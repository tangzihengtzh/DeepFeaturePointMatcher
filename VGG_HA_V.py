import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Mymodels import CustomVGG16NoPooling
from tools import preprocess, resize_tensors
from tmp_ha import select_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CustomVGG16NoPooling()
model.load_state_dict(torch.load('saved_pt/vgg16_cov.pth'))
model = model.to(device)

# 图像路径
image_path_A = r"data\demo1\resized\im0.png"
image_path_B = r"data\demo1\resized\im1.png"


# template, template_center_A, template_region_A = select_template(image_path_A)

from tools import split_image_into_regions
M = 18  # 横向分割成3块
N = 12  # 纵向分割成3块
region_size = (100, 100)  # 每个区域的尺寸 (宽, 高)
edge_percentage = 10  # 忽略边缘的10%

regions = split_image_into_regions(image_path_A, M, N, region_size, edge_percentage)

for idx, (template, center, region) in enumerate(regions):

    print(idx,"/",M*N)

    template, template_center_A, template_region_A = template, center, region

    print("Template center in A:", template_center_A)
    print("Template region in A:", template_region_A)

    # 转换模板为tensor
    template = Image.fromarray(template)
    template_tensor = preprocess(template).unsqueeze(0).to(device)

    # 读取B图像并进行预处理
    image_B = cv2.imread(image_path_B)
    height, width = image_B.shape[:2]

    # 获取模板的纵坐标范围（y1, y2）并计算对应的横向区域
    y1, y2 = template_region_A[1], template_region_A[3]  # A图中模板的纵坐标范围

    # 扩展10%的高度
    expansion = int(0.4 * (y2 - y1))  # 计算上下扩展的高度
    y1_expanded = max(0, y1 - expansion)  # 向上扩展，确保不超出上边界
    y2_expanded = min(height, y2 + expansion)  # 向下扩展，确保不超出下边界

    # 截取B图的横向区域，纵坐标范围上下扩展
    region_B_expanded = image_B[y1_expanded:y2_expanded, :]  # 提取B图中扩展后的区域

    # 将截取的区域转化为PIL图像，并进行预处理
    region_B_expanded_image = Image.fromarray(region_B_expanded)
    input_tensor_B = preprocess(region_B_expanded_image).unsqueeze(0).to(device)

    # 调整输入图像和模板的大小
    input_tensor_B, template_tensor, height_scale, width_scale = resize_tensors(input_tensor_B, template_tensor, target_size=448)

    with torch.no_grad():
        output_B = model(input_tensor_B.to(device))
        output_template = model(template_tensor)

        # 计算卷积结果
        kernel_size = output_template.shape[-1]
        padding = (kernel_size - 1)

        # 进行填充
        from torch import nn
        reflection_pad = nn.ConstantPad2d(padding=(padding, padding, padding, padding), value=-1)
        output_B_padded = reflection_pad(output_B)
        convolution_result = F.conv2d(output_B_padded, output_template)

        # 将卷积结果转回 CPU
        convolution_result = convolution_result.squeeze().cpu().numpy()

    # 创建与B图大小相同的全零矩阵
    convolution_result_full = np.zeros((height, width))
    print("Shape of full convolution result:", convolution_result_full.shape)

    # 获取卷积结果的尺寸
    convolution_height, convolution_width = convolution_result.shape

    # 计算需要的纵向和横向缩放比例
    target_height = y2_expanded - y1_expanded  # 填充区域的高度
    target_width = width  # B图的宽度，保持横向一致

    # 缩放卷积结果
    resized_convolution_result = cv2.resize(convolution_result, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 计算卷积结果填充的纵坐标范围
    y_start = y1_expanded
    y_end = min(y_start + resized_convolution_result.shape[0], y2_expanded)

    # 将缩放后的卷积结果填充到全零矩阵中
    convolution_result_full[y_start:y_end, :] = resized_convolution_result

    # 找到最大响应点
    from tmp_ha import find_max_loc_with_border_exclusion
    max_loc = find_max_loc_with_border_exclusion(convolution_result_full, exclusion_percent=0.05)
    match_location = (max_loc[1], max_loc[0])  # 转换为B图上的坐标

    # 获取A和B图的角点
    image_A = cv2.imread(image_path_A)
    image_B = cv2.imread(image_path_B)

    from tools import find_best_match
    template_region_A
    x1, y1, x2, y2 = template_region_A
    x_match, y_match = match_location
    SB_region = (
        x_match - (x2 - x1) // 2, y_match - (y2 - y1) // 2, x_match + (x2 - x1) // 2, y_match + (y2 - y1) // 2)

    corners_A,corners_B = find_best_match(image_path_A, image_path_B, template_region_A, SB_region)
    res = [corners_A,corners_B]
    with open('res_vgg_sift.txt', 'a') as file:
        # 格式化输出为字符串，并追加到文件末尾
        file.write(f"{res}\n")

    # print(SIFT_BEST)
    print(corners_A)
    print(corners_B)

    from tmp_ha import visualize_results
    # depth = visualize_results(image_A, template_center_A, image_B, match_location, corners_A, corners_B)








