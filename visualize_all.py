import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def visualize_matching_points(image_path_A, image_path_B, txt_path, threshold=0.1):
    """
    可视化匹配点，并排除纵坐标差异较大的匹配点。

    参数：
    image_path_A (str): 图像A的路径
    image_path_B (str): 图像B的路径
    txt_path (str): 存储匹配点的txt文件路径
    threshold (float): 纵坐标差异阈值，默认0.1表示10%的差异
    """
    # 读取图像A和B
    imgA = cv2.imread(image_path_A)
    imgB = cv2.imread(image_path_B)

    # 读取匹配点
    with open(txt_path, 'r') as f:
        matches = f.readlines()

    # 解析匹配点
    points_A = []
    points_B = []
    for match in matches:
        # 去掉换行符和多余空格
        match = match.strip()
        # 解析坐标
        match = match[1:-1]  # 去掉括号
        point_A, point_B = match.split('), (')
        point_A = point_A[1:]
        point_B = point_B[:-1]
        point_A = tuple(map(int, point_A.split(', ')))
        point_B = tuple(map(int, point_B.split(', ')))

        points_A.append(point_A)
        points_B.append(point_B)

    # 设置纵坐标差异阈值
    filtered_points_A = []
    filtered_points_B = []

    # 遍历所有匹配点，排除纵坐标差异较大的点
    for i in range(len(points_A)):
        ptA = points_A[i]
        ptB = points_B[i]

        # 计算纵坐标差异
        y_diff = abs(ptA[1] - ptB[1])

        # 计算纵坐标差异占总高度的百分比
        max_height = max(imgA.shape[0], imgB.shape[0])
        y_diff_percentage = y_diff / max_height

        # 如果纵坐标差异占比小于阈值，保留该点
        if y_diff_percentage < threshold:
            filtered_points_A.append(ptA)
            filtered_points_B.append(ptB)

    # 拼接图像
    result_img = np.hstack((imgA, imgB))

    # 绘制匹配点并连接
    for i in range(0,len(filtered_points_A)):
    # for i in range(5,15):
        ptA = filtered_points_A[i]
        ptB = filtered_points_B[i]

        # 将图B中的点的x坐标加上图A的宽度，便于在拼接图像中显示
        ptB = (ptB[0] + imgA.shape[1], ptB[1])

        # 绘制匹配点
        cv2.circle(result_img, ptA, 1, (0, 255, 0), 2)  # 图A中的点
        cv2.circle(result_img, ptB, 1, (0, 0, 255), 2)  # 图B中的点
        text_vgg = "P"+ str(random.randint(0, 255))

        # 连接匹配点
        cv2.line(result_img, ptA, ptB, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.putText(result_img, text_vgg, ptA, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(result_img, text_vgg, ptB, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.imshow(result_img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 示例调用：
# 图像路径
image_path_A = r"data\demo1\resized\im0.png"
image_path_B = r"data\demo1\resized\im1.png"
txt_path = 'res_vgg_sift.txt'  # 假设匹配点存储在这个txt文件中

# 设置纵坐标差异阈值，例如0.1（即10%的差异）
visualize_matching_points(image_path_A, image_path_B, txt_path, threshold=0.05)
