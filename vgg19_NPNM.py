import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


def sample_channels(tensor: torch.Tensor, N: int) -> torch.Tensor:
    """
    在通道维度上对输入的 tensor 进行抽样，以减少计算量。

    Args:
        tensor (torch.Tensor): 输入的 tensor，形状为 [bz, ch, H, W]。
        N (int): 抽样因子。输出的通道数将减少为原来的 1/N。

    Returns:
        torch.Tensor: 抽样后的 tensor，形状为 [bz, ch//N, H, W]。

    Raises:
        ValueError: 如果通道数不能被 N 整除。
    """
    if tensor.dim() != 4:
        raise ValueError(f"输入的 tensor 必须是 4 维的，但得到的是 {tensor.dim()} 维。")

    bz, ch, H, W = tensor.shape

    if ch % N != 0:
        raise ValueError(f"通道数 {ch} 不能被抽样因子 {N} 整除。")

    # 使用步长为 N 进行抽样
    sampled_tensor = tensor[:, ::N, :, :]

    return sampled_tensor

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

# 定义去除池化层和分类部分的VGG19自定义网络
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

        # 第三段卷积（无池化），包含4个卷积层
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第四段卷积（无池化），包含4个卷积层
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第五段卷积（无池化），包含4个卷积层
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        ws = 0.84

        x_c = x

        # 第一段卷积
        x = self.features1(x)
        x = self.features2(x)

        # 第二段卷积
        x = self.features3(x)
        x = self.features4(x)

        x = self.features5(x)

        x = sample_channels(x, 2)

        # 另一份输入
        x_c0= self.features1(x_c)
        x_c0 = nn.functional.max_pool2d(x_c0,2,2)
        x_c1 = self.features2(x_c0)
        x_c1 = nn.functional.max_pool2d(x_c1, 2, 2)
        x_c2 = self.features3(x_c1)

        x_c2 = nn.functional.max_pool2d(x_c2, 2, 2)
        x_c3 = self.features4(x_c2)

        print("rus:", x.shape, x_c.shape)

        we_s = 0.8
        print(x.shape,x_c.shape)

        # 合并两个特征
        # out1 = resize_and_concatenate(x_c0, x_c1, weight_large=we_s, weight_small=1-we_s)
        # out2 = resize_and_concatenate(out1, x_c2, weight_large=we_s, weight_small=1 - we_s)
        # out3 = resize_and_concatenate(x_c2, x_c3, weight_large=0.5, weight_small=0.5)
        # out3 = sample_channels(out3, 2)

        fL = resize_and_concatenate(x_c3, x_c2, weight_large=1, weight_small=1)

        out = resize_and_concatenate(x, fL, weight_large=we_s, weight_small=1 - we_s)


        print(out.shape)

        return out

    # def forward(self, x):
    #     x = self.features1(x)
    #     x = self.features2(x)
    #     x = self.features3(x)
    #     x = self.features4(x)
    #     x = self.features5(x)
    #     return x

def main():
    model_t = CustomVGG19NoPooling().cuda()
    model_t.load_state_dict(torch.load('custom_vgg19_no_pooling_no_classifier.pth'))

    input_tensor = torch.rand(1,3,10,10).cuda()

    # 执行前向传播
    output = model_t(input_tensor)

    # 打印输出的形状
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()