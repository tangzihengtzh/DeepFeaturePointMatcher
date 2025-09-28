import cv2
import os
import numpy as np


def edge_detection_and_overwrite(directory):
    # 获取目录下所有的jpg文件
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            # 构造完整的文件路径
            file_path = os.path.join(directory, filename)

            # 读取图像
            image = cv2.imread(file_path)

            if image is None:
                print(f"无法读取图像: {file_path}")
                continue

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用Canny算法进行边缘检测
            edges = cv2.Canny(gray, 100, 200)

            # 将单通道的边缘图像转换为三通道图像
            edges_rgb = cv2.merge([edges, edges, edges])

            # 保存边缘检测结果，直接覆盖原文件
            cv2.imwrite(file_path, edges_rgb)
            print(f"已覆盖文件: {file_path}")


# 调用函数，传入目标目录路径
directory_path = r"E:\python_prj\VGG16_CNT\demo_img\len_mea"  # 替换为实际目录路径
edge_detection_and_overwrite(directory_path)
