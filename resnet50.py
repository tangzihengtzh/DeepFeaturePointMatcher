import torch
import torch.nn as nn
from torchvision import models

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

# 初始化自定义网络
model = CustomVGG19NoPooling()

# 加载预训练的VGG19模型
pretrained_vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

for i in range(0,100):
    print(i,":",pretrained_vgg19.features[i])

# 将预训练的VGG19模型的权重导入自定义网络
with torch.no_grad():
    # 导入第一段卷积层权重
    model.features1[0].weight.data = pretrained_vgg19.features[0].weight.data.clone()
    model.features1[0].bias.data = pretrained_vgg19.features[0].bias.data.clone()
    model.features1[2].weight.data = pretrained_vgg19.features[2].weight.data.clone()
    model.features1[2].bias.data = pretrained_vgg19.features[2].bias.data.clone()

    # 导入第二段卷积层权重
    model.features2[0].weight.data = pretrained_vgg19.features[5].weight.data.clone()
    model.features2[0].bias.data = pretrained_vgg19.features[5].bias.data.clone()
    model.features2[2].weight.data = pretrained_vgg19.features[7].weight.data.clone()
    model.features2[2].bias.data = pretrained_vgg19.features[7].bias.data.clone()

    # 导入第三段卷积层权重
    model.features3[0].weight.data = pretrained_vgg19.features[10].weight.data.clone()
    model.features3[0].bias.data = pretrained_vgg19.features[10].bias.data.clone()
    model.features3[2].weight.data = pretrained_vgg19.features[12].weight.data.clone()
    model.features3[2].bias.data = pretrained_vgg19.features[12].bias.data.clone()
    model.features3[4].weight.data = pretrained_vgg19.features[14].weight.data.clone()
    model.features3[4].bias.data = pretrained_vgg19.features[14].bias.data.clone()

    # 导入第四段卷积层权重
    model.features4[0].weight.data = pretrained_vgg19.features[17].weight.data.clone()
    model.features4[0].bias.data = pretrained_vgg19.features[17].bias.data.clone()
    model.features4[2].weight.data = pretrained_vgg19.features[19].weight.data.clone()
    model.features4[2].bias.data = pretrained_vgg19.features[19].bias.data.clone()
    model.features4[4].weight.data = pretrained_vgg19.features[21].weight.data.clone()
    model.features4[4].bias.data = pretrained_vgg19.features[21].bias.data.clone()

    # 导入第五段卷积层权重
    model.features5[0].weight.data = pretrained_vgg19.features[24].weight.data.clone()
    model.features5[0].bias.data = pretrained_vgg19.features[24].bias.data.clone()
    # model.features5[2].weight.data = pretrained_vgg19.features[26].weight.data.clone()
    # model.features5[2].bias.data = pretrained_vgg19.features[26].bias.data.clone()
    model.features5[4].weight.data = pretrained_vgg19.features[28].weight.data.clone()
    model.features5[4].bias.data = pretrained_vgg19.features[28].bias.data.clone()

# 保存自定义网络的权重
torch.save(model.state_dict(), 'custom_vgg19_no_pooling_no_classifier.pth')
model_t = CustomVGG19NoPooling()
model_t.load_state_dict(torch.load('custom_vgg19_no_pooling_no_classifier.pth'))

print("Model weights saved to custom_vgg19_no_pooling_no_classifier.pth")
