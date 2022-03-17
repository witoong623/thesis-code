import cv2
import os
import unittest
from enum import Flag, auto

class UndistortType(Flag):
    BASIC = auto()
    REFINE = auto()
    REMAP = auto()
    CROP = auto()

class Undistorter:
    def __init__(self, calibration_file):
        self.calibration_file = calibration_file
        # TODO: delete this condition
        if not os.path.exists(self.calibration_file):
            self.undistort_image = lambda frame, flag: frame
            return

        self._read_file()

        self.new_cam_mat, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_mat, \
            self.dist_coeff, self.img_size, 1, self.img_size)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.cam_mat, self.dist_coeff, None, self.new_cam_mat, self.img_size, 5)

        self.width = self.roi[2]
        self.height = self.roi[3]

    def _read_file(self):
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_READ)
        self.img_size = tuple(fs.getNode('img_size').mat().astype(int).reshape(-1))
        self.cam_mat = fs.getNode('cam_mat').mat()
        self.dist_coeff = fs.getNode('dist_coeff').mat()

        fs.release()

    def undistort_image(self, frame, undistort_type=UndistortType.BASIC):
        ''' Undistort image using `getOptimalNewCameraMatrix`, `initUndistortRectifyMap` and `remap` functions. '''
        if undistort_type & UndistortType.BASIC:
            ret = cv2.undistort(frame, self.cam_mat, self.dist_coeff)
        elif undistort_type & UndistortType.REFINE:
            ret = cv2.undistort(frame, self.cam_mat, self.dist_coeff, None, self.new_cam_mat)
        elif undistort_type & UndistortType.REMAP:
            ret = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

        if undistort_type & UndistortType.CROP:
            x, y, w, h = self.roi
            ret = ret[y:y+h, x:x+w]

        H, W , C = ret.shape
        assert H == self.height and W == self.width

        return ret


class TestUndistorter(unittest.TestCase):
    CALIBRATE_FILE = 'robot_calibrate.yml'

    def setUp(self):
        self.undistorter = Undistorter(self.CALIBRATE_FILE)
        
        return super().setUp()

    def test_get_size(self):
        true_size = (1920, 1080)

        self.assertEqual(true_size, self.undistorter.img_size)

    def test_enum_str(self):
        enum_val = UndistortType.REMAP_CROP

        self.assertTrue(enum_val.value.startswith('REMAP'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
