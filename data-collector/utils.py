import cv2
import os

CURRENT_DIR = os.getcwd()

def save_image(image, name, dir):
    image_dir = os.path.join(CURRENT_DIR, dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_path = os.path.join(image_dir, name)
    cv2.imwrite(image_path, image)
