import cv2
import numpy as np

# 存储用户点击的点
clicked_points = []

# 鼠标回调函数：记录用户点击的点
def click_event(event, x, y, flags, params):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # 绘制点击点
        cv2.circle(img1_copy, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.imshow("Image", img1_copy)

# SIFT Demo 函数
def sift_demo(image_path_1, image_path_2):
    global img1, img1_copy, img2

    # 读取两张图片
    img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
    img1_copy = cv2.imread(image_path_1)  # 用于显示用户点击时的更新

    # 显示图片并让用户点击选择点
    cv2.imshow("Image", img1_copy)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)  # 等待用户点击
    cv2.destroyAllWindows()

    if len(clicked_points) < 2:
        print("请至少选择两个点进行匹配。")
        return

    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()

    # 检测特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用暴力匹配器进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据匹配的结果按距离排序
    matches = sorted(matches, key = lambda x: x.distance)

    # 找出用户点击的点对应的特征点
    selected_kp1 = []
    selected_kp2 = []
    good_matches = []

    for point in clicked_points:
        # 在第一张图片中寻找最接近用户点击的特征点
        min_dist1 = float('inf')
        closest_kp1 = None
        for kp in kp1:
            dist = np.sqrt((point[0] - kp.pt[0])**2 + (point[1] - kp.pt[1])**2)
            if dist < min_dist1:
                min_dist1 = dist
                closest_kp1 = kp
        selected_kp1.append(closest_kp1)

        # 在第二张图片中寻找与第一张图片匹配的特征点
        min_dist2 = float('inf')
        closest_kp2 = None
        for kp in kp2:
            dist = np.sqrt((point[0] - kp.pt[0])**2 + (point[1] - kp.pt[1])**2)
            if dist < min_dist2:
                min_dist2 = dist
                closest_kp2 = kp
        selected_kp2.append(closest_kp2)

    # 创建 DMatch 对象，将 selected_kp1 和 selected_kp2 中的特征点匹配起来
    for i in range(len(selected_kp1)):
        # 创建 DMatch 对象，queryIdx 和 trainIdx 分别是匹配点在两张图片中的索引
        match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=np.linalg.norm(np.array(selected_kp1[i].pt) - np.array(selected_kp2[i].pt)))
        good_matches.append(match)

    # 将匹配点绘制在图像上
    img_matches = cv2.drawMatches(img1, selected_kp1, img2, selected_kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配结果
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # 调用示例
# sift_demo('image1.jpg', 'image2.jpg')
#


# 调用示例
image_path_A = r'E:\python_prj\VGG16_CNT\demo_img\len_mea\im0.jpg'
image_path_B = r'E:\python_prj\VGG16_CNT\demo_img\len_mea\im1.jpg'
sift_demo(image_path_A, image_path_B)
