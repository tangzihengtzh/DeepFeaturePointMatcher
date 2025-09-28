import torch
import torch.nn as nn
from torchvision import models
import Mymodels
from Mymodels import resize_and_concatenate


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

        # # 自注意力模块
        # self.attention = SelfAttention(512)
        self.ws = 0.5

    def forward(self, x):
        x_c = x

        x = self.features1(x)
        x = self.features2(x)

        # visualize_feature_maps(x)
        # exit(5)

        x = self.features3(x)
        x = self.features4(x)

        # 此处添加
        # x = self.features5(x)


        x_c = self.features1(x_c)
        x_c = nn.functional.max_pool2d(x_c,2,2)
        x_c = self.features2(x_c)
        x_c = nn.functional.max_pool2d(x_c, 2, 2)
        x_c = self.features3(x_c)


        print("rus:",x.shape,x_c.shape)

        we_s = self.ws
        we_s = 0.0

        out = resize_and_concatenate(x,x_c,weight_large=we_s, weight_small=1-we_s)

        # print("f_s1,f_s2,f_s3,x",f_s1.shape,f_s2.shape,f_s3.shape,x.shape)
        # print(f"[f_s1: {f_s1.shape}, \nf_s2: {f_s2.shape}, \nf_s3: {f_s3.shape}, \nx: {x.shape}]")

        # x = nn.functional.max_pool2d(x, 2, 2)

        # x = self.features5(x)
        return out


# 只有在直接运行该脚本时才会执行测试代码
if __name__ == "__main__":
    # 初始化一个30x30大小的图片张量
    input_image = torch.randn(1, 3, 30, 30)  # batch_size = 1, channels = 3, height = 30, width = 30

    # 实例化修改后的VGG16模型
    model = CustomVGG16NoPooling()

    # 测试前向传播
    output = model(input_image)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
