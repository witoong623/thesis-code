import argparse
import cv2
import glob
import os
import re

from undistorter import UndistortType, Undistorter

FILE_PATTERN = re.compile(r'^.*\/|.*\\?[\w-]*video-(\d+).mp4')
CALIBRATE_FILE = 'dashcam_calibrate.yml'
# target capture device properties
HEIGHT = 1080
WIDTH = 1920
FPS = 10.0


def get_filepath(output_dir):
    video_files = glob.iglob(os.path.join(output_dir, '*.mp4'))
    matches = filter(None, map(FILE_PATTERN.match, video_files))
    latest_file = max(matches, key=lambda match: int(match.group(1)), default=None)

    if latest_file is None:
        index = 1
    else:
        index = int(latest_file.group(1)) + 1

    return os.path.join(output_dir, f'video-{index:03d}.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=0, help='id of video capturing device.')
    parser.add_argument('-o', '--output-dir', type=str, help='output directory path')
    args = parser.parse_args()

    # for calibrating camera
    undistort = Undistorter(CALIBRATE_FILE)

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError('Unable to open camera')

    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if original_width > WIDTH and original_height > HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print(f'original camera FPS = {cap.get(cv2.CAP_PROP_FPS)}')
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if args.output_dir is not None:
        video_dir = os.pardir.join(os.getcwd(), 'video-files')
    else:
        video_dir = args.output_dir

    video_writer = cv2.VideoWriter(get_filepath(video_dir), -1, FPS, (undistort.width, undistort.height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = undistort.undistort_image(frame, UndistortType.REFINE | UndistortType.CROP)

        cv2.imshow('Live', undistorted_frame)

        video_writer.write(undistorted_frame)

        iKey = cv2.waitKey(5)
        if iKey == ord('q') or iKey == ord('Q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
