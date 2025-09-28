from PIL import Image, ImageFilter
import os


def sharpen_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            # 构造图像文件的完整路径
            file_path = os.path.join(folder_path, filename)

            # 打开图像
            try:
                img = Image.open(file_path)

                # 应用锐化滤镜
                sharpened_img = img.filter(ImageFilter.SHARPEN)

                # 覆盖保存处理后的图像
                sharpened_img.save(file_path)
                print(f"已处理并覆盖：{filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 指定要处理的文件夹路径
folder_path = r"E:\python_prj\VGG16_CNT\demo_img\len_mea"
sharpen_images_in_folder(folder_path)
