import cv2
import numpy as np


def find_best_match(image_path_A, image_path_B, region_A, region_B):
    """
    找到两幅图像指定区域内匹配度最大的两个特征点坐标。

    参数:
    image_path_A (str): 图像A的路径
    image_path_B (str): 图像B的路径
    region_A (tuple): 图像A中指定的区域 (x1, y1, x2, y2)
    region_B (tuple): 图像B中指定的区域 (x1, y1, x2, y2)

    返回:
    tuple: 两个匹配点的坐标 (ptA, ptB)
    """

    # 读取图片
    imgA = cv2.imread(image_path_A, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(image_path_B, cv2.IMREAD_GRAYSCALE)

    # 提取指定区域
    x1_A, y1_A, x2_A, y2_A = region_A
    x1_B, y1_B, x2_B, y2_B = region_B
    imgA_region = imgA[y1_A:y2_A, x1_A:x2_A]
    imgB_region = imgB[y1_B:y2_B, x1_B:x2_B]

    # 使用SIFT进行特征点检测和描述符计算
    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(imgA_region, None)
    kpB, desB = sift.detectAndCompute(imgB_region, None)

    # 使用暴力匹配器进行匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desA, desB)

    # 按照匹配距离排序，选择距离最小的匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取最匹配的两个特征点
    best_match_A = kpA[matches[0].queryIdx].pt
    best_match_B = kpB[matches[0].trainIdx].pt

    second_best_match_A = kpA[matches[1].queryIdx].pt
    second_best_match_B = kpB[matches[1].trainIdx].pt

    # 返回这两个匹配点的坐标
    return (best_match_A, best_match_B), (second_best_match_A, second_best_match_B)


# 示例调用：
image_path_A = r"data\demo1\resized\im0.png"
image_path_B = r"data\demo1\resized\im1.png"
region_A = (100, 100, 500, 500)  # (x1, y1, x2, y2) 格式
region_B = region_A

best_match, second_best_match = find_best_match(image_path_A, image_path_B, region_A, region_B)
print("Best match point in A:", best_match[0])
print("Best match point in B:", best_match[1])
print("Second best match point in A:", second_best_match[0])
print("Second best match point in B:", second_best_match[1])

# 可视化匹配结果
imgA = cv2.imread(image_path_A)
imgB = cv2.imread(image_path_B)

# 拼接图像
result_img = np.hstack((imgA, imgB))

# 将匹配点画在拼接图像上
ptA1 = (int(best_match[0][0]), int(best_match[0][1]))
ptB1 = (int(second_best_match[0][0] + imgA.shape[1]), int(second_best_match[0][1]))

# ptA2 = (int(best_match[1][0]), int(best_match[1][1]))
# ptB2 = (int(second_best_match[1][0] + imgA.shape[1]), int(second_best_match[1][1]))

# 画出匹配的点
cv2.line(result_img, ptA1, ptB1, (0, 255, 0), 2)
# cv2.line(result_img, ptA2, ptB2, (0, 255, 0), 2)

# 显示结果
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.imshow(result_img)
plt.axis('off')  # 关闭坐标轴
plt.show()

