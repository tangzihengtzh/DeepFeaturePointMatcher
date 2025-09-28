import os
from PIL import Image


def resize_images_in_directory(directory, target_width):
    # 创建一个新的文件夹 'resized' 来存储调整后的图片
    resized_folder = os.path.join(directory, 'resized')
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    # 获取目录下所有的jpg文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            # 构建图片路径
            img_path = os.path.join(directory, filename)

            try:
                # 打开图片
                with Image.open(img_path) as img:
                    # 获取当前图片的宽高
                    width, height = img.size
                    # 计算新的高度，保持等比例缩放
                    target_height = int((target_width / width) * height)

                    # 调整图片大小
                    resized_img = img.resize((target_width, target_height))

                    # 保存调整后的图片到 'resized' 文件夹
                    resized_img.save(os.path.join(resized_folder, filename))
                    print(f"图片 {filename} 已调整大小并保存至 {os.path.join(resized_folder, filename)}")

            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")


# 示例用法
directory = r"data\demo11"  # 替换为你的目录路径
target_width = 1280  # 替换为你想要的横向分辨率
resize_images_in_directory(directory, target_width)
