import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch
from torch.nn import functional as F
# from tools import visualize_feature_maps


def resize_and_concatenate(tensor_large, tensor_small, weight_large=0.5, weight_small=0.5):
    """
    将较小的张量插值调整为与较大的张量相同的分辨率，并在通道维度上加权合并。

    参数:
    tensor_large (torch.Tensor): 更大的特征图张量。
    tensor_small (torch.Tensor): 更小的特征图张量。
    weight_large (float): 合并时较大张量的权重系数。
    weight_small (float): 合并时较小张量的权重系数。

    返回:
    torch.Tensor: 调整大小并加权合并后的张量。
    """
    # 获取大张量的空间尺寸（高度和宽度）
    size_large = tensor_large.shape[2:]  # 大张量的空间尺寸

    # 将较小的张量调整到较大张量的大小
    tensor_small_resized = F.interpolate(tensor_small, size=size_large, mode='bilinear', align_corners=False)

    # 在通道维度上加权合并
    result = torch.cat((weight_large * tensor_large, weight_small * tensor_small_resized), dim=1)

    return result

# 定义一个去掉池化层和分类部分的VGG19自定义网络
class CustomVGG19NoPooling(nn.Module):
    def __init__(self):
        super(CustomVGG19NoPooling, self).__init__()

        # 第一段卷积（无池化）
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第二段卷积（无池化）
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第三段卷积（无池化）
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第四段卷积（无池化）
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第五段卷积（无池化）
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x

# class CustomVGG19NoPooling(nn.Module):
#     def __init__(self):
#         super(CustomVGG19NoPooling, self).__init__()
#
#         # 第一段卷积（无池化）
#         self.features1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         # 第二段卷积（无池化）
#         self.features2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         # 第三段卷积（无池化）
#         self.features3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         # 第四段卷积（无池化）
#         self.features4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         # 第五段卷积（无池化）
#         self.features5 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#
#
#     def forward(self, x):
#
#         ws = 0.84
#
#         x_c = x
#
#         # 第一段卷积
#         x = self.features1(x)
#         x = self.features2(x)
#
#         # 第二段卷积
#         x = self.features3(x)
#         x = self.features4(x)
#
#         # 另一份输入
#         x_c = self.features1(x_c)
#         x_c = F.max_pool2d(x_c, 2, 2)
#         x_c = self.features2(x_c)
#         x_c = F.max_pool2d(x_c, 2, 2)
#         x_c = self.features3(x_c)
#
#         print("rus:", x.shape, x_c.shape)
#
#         we_s = self.ws
#
#         # 合并两个特征
#         out = resize_and_concatenate(x, x_c, weight_large=we_s, weight_small=1-we_s)
#
#         return out

from PIL import Image
import torchvision.transforms as transforms
import torch
# 加载图像并进行预处理
def load_image(image_path):
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')

    # 定义预处理操作：仅转换为Tensor，不改变分辨率
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为 PyTorch 的 Tensor 格式
    ])

    # 应用预处理并增加批次维度
    image_tensor = preprocess(image).unsqueeze(0)  # 增加批次维度，变为形状 (1, 3, H, W)

    # 将图像转移到GPU上（如果有可用的GPU）
    return image_tensor.cuda()



def main():
    # 初始化自定义的 VGG19 网络
    model = CustomVGG19NoPooling().cuda()
    model.load_state_dict(torch.load('custom_vgg19_no_pooling_no_classifier.pth'))

    # model = model.cuda()

    # 示例调用，加载图像等
    image_path = r"E:\python_prj\VGG16_CNT\demo_img\demo17\src.jpg"  # 替换为您的图像路径
    input_tensor = load_image(image_path).cuda()

    # 执行前向传播
    output = model(input_tensor)

    # 打印输出的形状
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
