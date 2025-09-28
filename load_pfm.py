import numpy as np
import matplotlib.pyplot as plt


def read_pfm(file):
    """读取 PFM 文件"""
    with open(file, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header != 'Pf' and header != 'PF':
            raise ValueError('Not a PFM file.')

        # 读取图像的宽度和高度
        width, height = map(int, f.readline().decode('ascii').split())

        # 读取标度因子
        scale = float(f.readline().decode('ascii').strip())

        # 读取图像数据
        data = np.fromfile(f, dtype='<f4')  # '<f4' 表示小端格式的浮动点数

        # 将数据重塑为图像
        image = np.reshape(data, (height, width, 3 if header == 'PF' else 1))  # 彩色或灰度

        # 如果标度因子小于零，翻转图像
        if scale < 0:
            image = np.flipud(image)

        # 如果是彩色图像，将其转换为灰度图像（通常深度图是单通道的）
        if header == 'PF':
            image = np.mean(image, axis=2)

        return image


def visualize_depth_map(depth_map):
    """可视化深度图"""
    plt.imshow(depth_map, cmap='inferno')  # 使用' inferno' 色图
    plt.colorbar(label='Depth (mm)')  # 添加颜色条来显示深度值
    plt.title('Depth Map')
    plt.show()


# 使用例子
file_path = r"E:\ML_DATA\all\data\artroom1\disp1.pfm"
depth_map = read_pfm(file_path)
visualize_depth_map(depth_map)
