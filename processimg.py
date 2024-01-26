import os
import random
from PIL import Image

def process_images(directory, new_directory):
    os.makedirs(new_directory, exist_ok=True)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                # 移动图片
                x_shift = random.randint(0, 4)
                y_shift = random.randint(0, 4)
                img = img.transform(img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))

                # 调整颜色通道
                img = img.convert("RGB")
                r, g, b = img.split()

                rchange = random.randint(-5, 5)
                gchange = random.randint(-5, 5)
                bchange = random.randint(-5, 5)

                r = r.point(lambda i: i + rchange)
                g = g.point(lambda i: i + gchange)
                b = b.point(lambda i: i + bchange)

                img = Image.merge("RGB", (r, g, b))
                
                # 保存图片
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_sft_{x_shift}_{y_shift}_clr_{rchange}_{gchange}_{bchange}{ext}"
                new_path = os.path.join(new_directory, new_filename)
                img.save(new_path)
# Example usage
process_images(r"D:\hust\MediaLab\IAD\Anomalib\patchcore-inspection\data\MVTec\fu\train\good", r"D:\hust\MediaLab\IAD\Anomalib\patchcore-inspection\data\MVTec\fu\train\good_processed")