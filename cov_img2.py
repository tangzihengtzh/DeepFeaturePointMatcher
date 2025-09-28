import math
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import cv2
from Mymodels import CustomVGG16NoPooling
from tools import visualize_results
from tools import resize_tensors
from tools import preprocess
import os

k = 13


def main(image_path_s, template_path_s, image_path_or):
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_image(main_image_path):
        main_image = cv2.imread(main_image_path)
        main_image_resized = transform(main_image).unsqueeze(0).cuda()  # 如果使用GPU，请启用
        print("main_image_resized:", main_image_resized.shape)
        return main_image_resized

    # 初始化自定义网络
    model = CustomVGG16NoPooling()

    # 加载保存的权重
    model.load_state_dict(torch.load('saved_pt/vgg16_cov.pth'))
    print("Weights successfully loaded into the custom model.")

    image_path = image_path_or
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)  # 增加一个批量维度

    # 初始化模板路径
    template_path = template_path_s

    # 初始化平均张量
    avr_tensor = None
    count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 移动张量到设备
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # 读取并预处理模板
    template = Image.open(template_path)
    template_tensor = preprocess(template).unsqueeze(0).to(device)

    width, height = image.size
    area1 = width * height
    width, height = template.size
    area2 = width * height
    model.ws = k * (1 - math.sqrt(area2 / area1))
    print(model.ws)

    print("调整前：", input_tensor.shape, template_tensor.shape)
    input_tensor, template_tensor, height_scale, width_scale = resize_tensors(input_tensor, template_tensor,
                                                                              target_size=448)
    print("调整后：", input_tensor.shape, template_tensor.shape)

    # 前向传播并获取结果
    with torch.no_grad():
        start_time = time.time()
        output_img = model(input_tensor.to(device))
        print("原图特征图：", output_img.shape)

        output_tem = model(template_tensor)
        print("模板特征图：", output_tem.shape)
        kernel_size = output_tem.shape[-1]
        padding = (kernel_size - 1)
        convolution_result = F.conv2d(output_img, output_tem, padding=padding)
        end_time = time.time()

        print("卷积结果", convolution_result.shape)

        # 计算并打印运行时间
        elapsed_time = end_time - start_time
        print(f"using time：{elapsed_time} s")

        # 获取最小高度和宽度
        min_height = min(avr_tensor.shape[2] if avr_tensor is not None else convolution_result.shape[2],
                         convolution_result.shape[2])
        min_width = min(avr_tensor.shape[3] if avr_tensor is not None else convolution_result.shape[3],
                        convolution_result.shape[3])

        # 裁剪卷积结果到最小尺寸
        convolution_result = convolution_result[:, :, :min_height, :min_width]
        print("moving the data from GPU to CPU ... ")
        # 将张量从 GPU 移动到 CPU
        convolution_result = convolution_result.to("cpu")
        print("moving the data from GPU to CPU ... finish ")

        # 转换为 NumPy 数组
        convolution_result_np = convolution_result.numpy()

        if avr_tensor is None:
            avr_tensor = convolution_result
        else:
            avr_tensor = avr_tensor[:, :, :min_height, :min_width]  # 同样裁剪avr_tensor
            avr_tensor += convolution_result

        count += 1

    # 计算平均张量
    avr_tensor /= count
    print("对比度：", k)

    des_np = visualize_results(avr_tensor, image_path, kernel_size, k, [template_path])
    print(des_np.shape)

    from tools import otsu_threshold_and_visualize
    otsu_threshold_and_visualize(des_np)


root = r'E:\python_prj\VGG16_CNT\demo_img\len_mea'
# 获取root目录下所有的jpg文件
jpg_files = [file for file in os.listdir(root) if file.endswith(".jpg")]

# 检查是否找到两个jpg文件
if len(jpg_files) != 2:
    raise ValueError("根目录下必须包含两个jpg文件！")

# 获取文件的完整路径
imgpath = os.path.join(root, jpg_files[0])
tmppath = os.path.join(root, jpg_files[1])

# 根据文件大小来区分大图和小图
if os.path.getsize(imgpath) < os.path.getsize(tmppath):
    imgpath, tmppath = tmppath, imgpath  # 交换路径，使 imgpath 指向大图，tmppath 指向小图

k = 1 / 1.2
main(root, tmppath, imgpath)
