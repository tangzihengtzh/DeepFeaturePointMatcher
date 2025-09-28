import torch
import matplotlib.pyplot as plt


def plot_grayscale_image(tensor):
    """
    输入一个单层灰度图的Tensor，并将其绘制为灰度图。

    :param tensor: 输入的单层Tensor，尺寸应为 (H, W)。
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("输入必须是一个Tensor类型")

    # 检查输入tensor的形状是否是2D（即单层灰度图）
    if tensor.dim() != 2:
        raise ValueError("输入的tensor应该是一个二维灰度图（H, W）")

    # 将tensor转化为numpy数组，便于matplotlib处理
    img = tensor.numpy()

    # 使用matplotlib绘制灰度图
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 测试代码
if __name__ == "__main__":
    # 创建一个随机的二维Tensor（灰度图）
    gray_tensor = torch.randn(30, 30)  # 30x30大小的灰度图

    # 调用绘图函数
    plot_grayscale_image(gray_tensor)
