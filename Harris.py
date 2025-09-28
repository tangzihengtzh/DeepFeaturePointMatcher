import cv2
import numpy as np

# 全局变量，用于存储框选区域的坐标
x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False  # 标志位，表示是否正在绘制矩形框


# 鼠标事件回调函数
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键时，记录起始点
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # 拖动鼠标时，更新终点
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x1, y1), (x, y), (0, 255, 0), 2)  # 绘制矩形框
            cv2.imshow('Harris Corner Detection - Draw Rectangle', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键松开时，确定矩形框
        drawing = False
        x2, y2 = x, y
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框
        cv2.imshow('Harris Corner Detection - Draw Rectangle', img)


# 读取图像
image_path = r'E:\python_prj\VGG16_CNT\demo_img\len_mea\im0.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 转换为浮点类型
gray = np.float32(gray)

# 设置鼠标回调函数
cv2.namedWindow('Harris Corner Detection - Draw Rectangle')
cv2.setMouseCallback('Harris Corner Detection - Draw Rectangle', draw_rectangle)

# 等待用户框选区域
print("请在图像上框选一个区域...")
cv2.imshow('Harris Corner Detection - Draw Rectangle', img)
cv2.waitKey(0)

# 检查是否框选了区域
if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
    # Harris角点检测
    dst = cv2.cornerHarris(gray, 4, 5, 0.04)

    # 提取指定区域的角点响应图
    roi = dst[y1:y2, x1:x2]

    # 找到最“好”的角点：即角点强度最大的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)

    # 显示最“好”的角点
    best_x, best_y = max_loc
    cv2.circle(img, (best_x + x1, best_y + y1), 1, (0, 255, 0), 2)  # 使用绿色圆圈标记角点

    # 显示角点检测结果
    cv2.imshow('Best Harris Corner', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("没有框选区域。")

cv2.destroyAllWindows()
