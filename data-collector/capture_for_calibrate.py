import argparse
import cv2
import glob
import os
import numpy as np

from utils import save_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-index', type=int, default=0, help='id of video capturing device.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("ERROR! Unable to open camera\n")
        exit(1)

    print("Start grabbing")
    print("Press s to save images and q to terminate")

    frameAdd = 0

    while True:
        _, frame = cap.read()
        if frame is None:
            print("ERROR! blank frame grabbed\n")
            exit(1)
        cv2.imshow("Live", frame)

        iKey = cv2.waitKey(5)
        if iKey == ord('s') or iKey == ord('S'):
            save_image(frame, name=f'{frameAdd}.jpg', dir='calibrated_images')
            frameAdd += 1
            print(f"Frame: {frameAdd} has been saved.")
        elif iKey == ord('q') or iKey == ord('Q'):
            break

    cap.release()
