import os
from PIL import Image

def convert_bmp_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(directory, filename)
            jpg_path = os.path.splitext(bmp_path)[0] + ".jpg"
            with Image.open(bmp_path) as img:
                img.save(jpg_path, "JPEG")

# Example usage:
convert_bmp_to_jpg(r"D:\hust\MediaLab\IAD\Anomalib\patchcore-inspection\data\MVTec\fu\test\ng")
