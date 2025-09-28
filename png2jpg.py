import os
from PIL import Image


def convert_to_rgb(directory):
    # 获取指定目录下所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 检查文件是否是.jpg或.png
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 转换为3通道的RGB模式
                    img_rgb = img.convert('RGB')
                    # 覆盖保存为JPEG格式
                    img_rgb.save(file_path, 'JPEG')
                    print(f"Converted and saved: {filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


# 指定需要转换的目录
directory = r"data\demo1"
convert_to_rgb(directory)
