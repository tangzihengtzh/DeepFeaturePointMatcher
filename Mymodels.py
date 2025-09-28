# Mymodels.py

import torch.nn as nn
from torchvision import models
import torch
from torch.nn import functional as F
# from tools import visualize_feature_maps

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 32, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 32, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Query, Key, Value 计算
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, N, C]
        key = self.key_conv(x).view(batch_size, -1, width * height)  # [B, C, N]
        value = self.value_conv(x).view(batch_size, -1, width * height)  # [B, C, N]

        # 注意力图计算
        print(query.shape, key.shape)
        attention = torch.bmm(query, key)  # [B, N, N]
        attention = F.softmax(attention, dim=-1)  # Softmax 归一化

        # 输出
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, width, height)

        # 使用可学习的参数 gamma 调节输出
        out = self.gamma * out + x
        return out

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

class CustomVGG16NoPooling(nn.Module):
    def __init__(self):
        super(CustomVGG16NoPooling, self).__init__()

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

        self.ws = 0.8

    def forward(self, x):
        x_c = x

        x = self.features1(x)
        x = self.features2(x)

        x = self.features3(x)
        x = self.features4(x)

        x_c = self.features1(x_c)
        x_c = nn.functional.max_pool2d(x_c,2,2)
        x_c = self.features2(x_c)
        x_c = nn.functional.max_pool2d(x_c, 2, 2)
        x_c = self.features3(x_c)

        print("rus:",x.shape,x_c.shape)

        we_s = self.ws

        out = resize_and_concatenate(x,x_c,weight_large=we_s, weight_small=1-we_s)

        return out

class CustomVGG16NoPooling_cov1(nn.Module):
    def __init__(self):
        super(CustomVGG16NoPooling_cov1, self).__init__()

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
        # x = self.features3(x)
        # x = self.features4(x)
        # x = self.features5(x)
        return x

import torch
import torch.nn as nn
from torchvision import models

# 定义一个去掉池化层和分类部分的ResNet50自定义网络
class CustomResNet50NoPooling(nn.Module):
    def __init__(self):
        super(CustomResNet50NoPooling, self).__init__()

        # 第一段卷积（无池化）
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 残差块1（无池化），增加downsample匹配通道
        self.features2 = nn.Sequential(
            models.resnet.Bottleneck(64, 64, stride=1, downsample=nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256))),
            models.resnet.Bottleneck(256, 64, stride=1),
            models.resnet.Bottleneck(256, 64, stride=1)
        )

        # 残差块2（无池化），增加downsample匹配通道
        self.features3 = nn.Sequential(
            models.resnet.Bottleneck(256, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512))),
            models.resnet.Bottleneck(512, 128, stride=1),
            models.resnet.Bottleneck(512, 128, stride=1),
            models.resnet.Bottleneck(512, 128, stride=1)
        )

        # 残差块3（无池化），增加downsample匹配通道
        self.features4 = nn.Sequential(
            models.resnet.Bottleneck(512, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(1024))),
            models.resnet.Bottleneck(1024, 256, stride=1),
            models.resnet.Bottleneck(1024, 256, stride=1),
            models.resnet.Bottleneck(1024, 256, stride=1),
            models.resnet.Bottleneck(1024, 256, stride=1),
            models.resnet.Bottleneck(1024, 256, stride=1)
        )

        # 残差块4（无池化），增加downsample匹配通道
        self.features5 = nn.Sequential(
            models.resnet.Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(2048))),
            models.resnet.Bottleneck(2048, 512, stride=1),
            models.resnet.Bottleneck(2048, 512, stride=1)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 =  nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        avg_out = nn.functional.relu(avg_out)
        max_out = self.fc2(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 下采样路径
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_relu0 = nn.ReLU(inplace=True)
        self.enc_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc_relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_relu2 = nn.ReLU(inplace=True)
        self.enc_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc_relu3 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间层
        self.bottleneck_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_relu4 = nn.ReLU(inplace=True)
        self.bottleneck_conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bottleneck_relu5 = nn.ReLU(inplace=True)

        # 上采样路径
        self.tem_conv1 = nn.Conv2d(9, 128, kernel_size=3, padding=1)
        self.ch1=ChannelAttention(128,128)
        self.upconv6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_relu6 = nn.ReLU(inplace=True)
        self.dec_conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_relu7 = nn.ReLU(inplace=True)

        self.tem_conv1 = nn.Conv2d(9, 128, kernel_size=3, padding=1)
        self.ch2=ChannelAttention(128,64)
        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_relu8 = nn.ReLU(inplace=True)
        self.dec_conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_relu9 = nn.ReLU(inplace=True)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self ,image):
        # 编码路径
        enc0 = self.enc_relu0(self.enc_conv0(image))
        enc1 = self.enc_relu1(self.enc_conv1(enc0))
        pool0 = self.pool0(enc1)
        # print("enc1.shape",enc1.shape)
        # exit(0x01)

        # plot_feature_maps(pool0)

        enc2 = self.enc_relu2(self.enc_conv2(pool0))
        enc3 = self.enc_relu3(self.enc_conv3(enc2))
        pool1 = self.pool1(enc3)
        # print("enc3.shape",enc3.shape)
        # exit(0x01)

        # 中间层
        bottleneck = self.bottleneck_relu4(self.bottleneck_conv4(pool1))
        bottleneck = self.bottleneck_relu5(self.bottleneck_conv5(bottleneck))
        # print("bottleneck.shape",bottleneck.shape)
        # exit(0x01)

        # 解码路径
        upconv6 = self.upconv6(bottleneck)
        concat6 = torch.cat((upconv6, enc3), dim=1)
        dec6 = self.dec_relu6(self.dec_conv6(concat6))
        dec7 = self.dec_relu7(self.dec_conv7(dec6))

        upconv7 = self.upconv7(dec7)
        concat7 = torch.cat((upconv7, enc1), dim=1)
        dec8 = self.dec_relu8(self.dec_conv8(concat7))
        dec9 = self.dec_relu9(self.dec_conv9(dec8))

        # 最终输出层
        output = self.final_conv(dec9)

        return output

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
    # 初始化自定义的 ResNet50 网络
    model = CustomVGG16NoPooling().cuda()
    # model.load_state_dict(torch.load('saved_pt/custom_resnet50_no_pooling.pth'))
    model.load_state_dict(torch.load('saved_pt/vgg16_cov.pth'))

    # 创建一个随机输入张量，模拟输入的图像
    # input_tensor = torch.randn(1, 3, 448, 448).cuda()  # Batch size: 1, Channels: 3, Height: 224, Width: 224
    # 示例调用
    image_path = r"E:\python_prj\VGG16_CNT\demo_img\demo17\src.jpg"  # 替换为你的 jpg 图像路径
    input_tensor = load_image(image_path)

    # 执行前向传播
    output = model(input_tensor)

    # 打印输出的形状
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()

