import cv2
import numpy as np
import os
import glob


IMG_PATH_PATTERN = os.getcwd() + os.sep + 'calibrated_images' + os.sep + '*.jpg'
CALIBRATE_FILE = 'dashcam_calibrate.yml'

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 7)
# scale of obj
OBJ_SCALE = 3.7


def save_camera_calibration(img_size, cam_mat, dist_coeff, rvecs, tvecs, rep_err, total_avg_err):
    fs = cv2.FileStorage(CALIBRATE_FILE, cv2.FILE_STORAGE_WRITE)
    fs.write('img_size', img_size)
    fs.write('cam_mat', cam_mat)
    fs.write('dist_coeff', dist_coeff)

    fs.write('rvecs_size', len(rvecs))
    for i, val in enumerate(rvecs, 1):
        fs.write(f'rvecs_{i}', val)

    fs.write('tvecs_size', len(tvecs))
    for i, val in enumerate(tvecs, 1):
        fs.write(f'tvecs_{i}', val)

    fs.write('rep_err', rep_err)
    fs.release()


if __name__ == '__main__':
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = (OBJ_SCALE * np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]).T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(IMG_PATH_PATTERN)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if found == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, found)
        
        # cv2.imshow('img',img)
        # cv2.waitKey(50)

    # cv2.destroyAllWindows()

    print(f'number of found pattern is {len(objpoints)}')

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    print(f'obj size is {len(imgpoints)}')
    print(f'imgpoints size is {len(imgpoints)}')
    rep_err, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(cam_mat)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    print(f'Reprojection error: {rep_err}')

    # Show images of undistorted
    # for fname in images:
    #     img = cv2.imread(fname)
    #     res = cv2.undistort(img, cam_mat, dist)
    #     cv2.imshow('img',img)
    #     cv2.imshow('res',res)
    #     cv2.waitKey(0)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mat, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    total_err = mean_error / len(objpoints)
    print(f"total error: {total_err}")

    save_camera_calibration(gray.shape[::-1], cam_mat, dist, rvecs, tvecs, rep_err, total_err)
