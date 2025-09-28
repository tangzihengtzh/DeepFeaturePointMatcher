import torch
import torch.nn as nn
from torchvision import models

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
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x

# 初始化自定义网络
model = CustomVGG19NoPooling()

# 加载预训练的VGG19模型
pretrained_vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# 将预训练的VGG19模型的权重导入自定义网络
with torch.no_grad():
    # 功能1
    model.features1[0].weight.copy_(pretrained_vgg19.features[0].weight)
    model.features1[0].bias.copy_(pretrained_vgg19.features[0].bias)
    model.features1[2].weight.copy_(pretrained_vgg19.features[2].weight)
    model.features1[2].bias.copy_(pretrained_vgg19.features[2].bias)

    # 功能2
    model.features2[0].weight.copy_(pretrained_vgg19.features[5].weight)
    model.features2[0].bias.copy_(pretrained_vgg19.features[5].bias)
    model.features2[2].weight.copy_(pretrained_vgg19.features[7].weight)
    model.features2[2].bias.copy_(pretrained_vgg19.features[7].bias)

    # 功能3
    model.features3[0].weight.copy_(pretrained_vgg19.features[10].weight)
    model.features3[0].bias.copy_(pretrained_vgg19.features[10].bias)
    model.features3[2].weight.copy_(pretrained_vgg19.features[12].weight)
    model.features3[2].bias.copy_(pretrained_vgg19.features[12].bias)
    model.features3[4].weight.copy_(pretrained_vgg19.features[14].weight)
    model.features3[4].bias.copy_(pretrained_vgg19.features[14].bias)
    model.features3[6].weight.copy_(pretrained_vgg19.features[16].weight)
    model.features3[6].bias.copy_(pretrained_vgg19.features[16].bias)

    # 功能4
    model.features4[0].weight.copy_(pretrained_vgg19.features[19].weight)
    model.features4[0].bias.copy_(pretrained_vgg19.features[19].bias)
    model.features4[2].weight.copy_(pretrained_vgg19.features[21].weight)
    model.features4[2].bias.copy_(pretrained_vgg19.features[21].bias)
    model.features4[4].weight.copy_(pretrained_vgg19.features[23].weight)
    model.features4[4].bias.copy_(pretrained_vgg19.features[23].bias)
    model.features4[6].weight.copy_(pretrained_vgg19.features[25].weight)
    model.features4[6].bias.copy_(pretrained_vgg19.features[25].bias)

    # 功能5
    model.features5[0].weight.copy_(pretrained_vgg19.features[28].weight)
    model.features5[0].bias.copy_(pretrained_vgg19.features[28].bias)
    model.features5[2].weight.copy_(pretrained_vgg19.features[30].weight)
    model.features5[2].bias.copy_(pretrained_vgg19.features[30].bias)
    model.features5[4].weight.copy_(pretrained_vgg19.features[32].weight)
    model.features5[4].bias.copy_(pretrained_vgg19.features[32].bias)
    model.features5[6].weight.copy_(pretrained_vgg19.features[34].weight)
    model.features5[6].bias.copy_(pretrained_vgg19.features[34].bias)

# 保存自定义网络的权重
torch.save(model.state_dict(), 'custom_vgg19_no_pooling_no_classifier.pth')

model_t = CustomVGG19NoPooling().cuda()
model_t.load_state_dict(torch.load('custom_vgg19_no_pooling_no_classifier.pth'))

input_tensor = torch.rand(3,10,10).cuda()

# 执行前向传播
output = model_t(input_tensor)

# 打印输出的形状
print("Output shape:", output.shape)

print("Model weights saved to custom_vgg19_no_pooling_no_classifier.pth")
