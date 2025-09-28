import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.nn import functional as F
import mytools
import Mymodels

class ModifiedVGG16(nn.Module):
    def __init__(self):
        super(ModifiedVGG16, self).__init__()

        # 加载预训练的VGG16模型
        vgg16 = models.vgg16(pretrained=True)

        # 去除VGG16中的分类器（fc层）和池化层
        features = list(vgg16.features.children())
        features = [layer for layer in features if not isinstance(layer, nn.MaxPool2d)]

        # 将特征层转为一个nn.Sequential模块
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


def extract_subimage(image, center, size=30):
    """
    从图像中提取一个大小为size的子图，子图的中心为center坐标
    :param image: 输入图像 (PIL图像)
    :param center: 中心坐标 (x, y)
    :param size: 子图大小，默认为30x30
    :return: 子图的Tensor
    """
    width, height = image.size
    x, y = center

    # 计算子图的边界
    half_size = size // 2
    left = max(x - half_size, 0)
    top = max(y - half_size, 0)
    right = min(x + half_size, width)
    bottom = min(y + half_size, height)

    # 裁剪子图并转化为Tensor
    subimage = image.crop((left, top, right, bottom))

    # 进行必要的转换，将PIL图像转为Tensor并归一化
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # 确保子图是30x30
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    subimage_tensor = transform(subimage).unsqueeze(0)  # 增加batch维度
    return subimage_tensor


def process_image_and_extract_feature(image_path, center,size = 30):
    # 实例化修改后的VGG16模型
    # model = ModifiedVGG16().cuda()
    # model.eval()  # 设置为评估模式

    # 打开图像并读取
    image = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB模式

    # 提取以给定坐标为中心的30x30子图
    subimage_tensor = extract_subimage(image, center, size = size).cuda()

    # 进行前向传播，获取卷积结果
    model = Mymodels.CustomVGG16NoPooling_cov1().cuda()
    # 加载保存的权重
    model.load_state_dict(torch.load('saved_pt/vgg16_cov.pth'))
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        feature_map = model(subimage_tensor)

    return feature_map


# 测试代码
if __name__ == "__main__":
    # 输入图像路径和坐标
    image0 = r"data\demo1\im0.png"
    image1 = r"data\demo1\im1.png"
    center = (100, 100)  # 例：以(50, 50)为中心，提取子图

    # 处理图像并提取卷积特征
    feature_map0 = process_image_and_extract_feature(image1, center,60)
    feature_map1 = process_image_and_extract_feature(image1, center, 1040)

    # 打印卷积结果的形状
    print("Feature map shape:", feature_map0.shape)
    print("Feature map shape:", feature_map1.shape)

    result =  F.conv2d(feature_map1,feature_map0)
    result = result.squeeze(0)
    result = result.squeeze(0)
    print(result.shape)
    result = result.cpu()
    mytools.plot_grayscale_image(result)


