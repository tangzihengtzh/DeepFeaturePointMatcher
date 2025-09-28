import numpy as np


def compute_depth(left_coord, right_coord, K_left, K_right, baseline):
    # 获取左图和右图的匹配点
    x_L, y_L = left_coord
    x_R, y_R = right_coord

    # 计算视差
    disparity = x_L - x_R  # 左图和右图的x坐标差

    # 提取焦距
    f_L = K_left[0, 0]  # 左图焦距 fx
    f_R = K_right[0, 0]  # 右图焦距 fx

    # 使用视差公式计算深度 Z = (f * B) / d
    if disparity == 0:
        raise ValueError("Disparity is zero, can't compute depth")

    depth = (f_L * baseline) / disparity
    return depth


# 示例：假设有以下数据
left_coord = (623, 383)  # 左图匹配点
right_coord = (573, 382) # 右图匹配点

# (509, 344), (466, 343)

# 假设左右相机的内参矩阵（焦距 f_x, f_y 和主点坐标 c_x, c_y）
K_left = np.array([[1733.74, 0, 792.27],
                   [0, 1733.74, 541.89],
                   [0, 0, 1]])

K_right = np.array([[1733.74, 0, 792.27],
                    [0, 1733.74, 541.89],
                    [0, 0, 1]])

# 基线距离（单位：毫米）
baseline = 536.62  # 例如，相机间的距离为 536.62 mm

# 计算深度
depth = compute_depth(left_coord, right_coord, K_left, K_right, baseline)*1280/1920
print("Estimated depth:", depth/1000)
