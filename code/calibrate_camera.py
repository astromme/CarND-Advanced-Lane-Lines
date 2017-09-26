import numpy as np
import cv2
import glob
import pickle

def calibrate_camera(images=glob.glob('camera_cal/calibration*.jpg')):
    """
    Computes the camera calibration matrix and distortion coefficients given a set of chessboard images.
    """

    # prepare object points
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #write_name = 'corners_found'+str(i)+'.jpg'
            #cv2.imwrite(write_name, img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def get_undistort(calibration_filename='calibration.pickle'):
    """
    Calibrates camera and returns function to undistort images.
    Loads calibration from calibration.pickle if it exists.
    Writes the calibration to calibration.pickle if it doesn't exist.
    """

    try:
        with open(calibration_filename, 'rb') as f:
            values = pickle.load(f)
            mtx = values['mtx']
            dist = values['dist']
            print('loaded calibration from {}'.format(calibration_filename))
    except (FileNotFoundError, EOFError) as e:
        print('calibrating camera')
        mtx, dist = calibrate_camera()

        with open(calibration_filename, 'wb') as f:
            pickle.dump({
                'mtx': mtx,
                'dist': dist,
            }, f)

    def undistort(image):
        return cv2.undistort(image, mtx, dist, None, mtx)

    return undistort

def write_test_image(input_filename='camera_cal/calibration1.jpg', output_filename='examples/undistorted_calibration1.jpg'):
    """
    undistorts an image and writes it to a file.
    """

    image = cv2.imread(input_filename)
    undistort = get_undistort()
    cv2.imwrite(output_filename, undistort(image))



if __name__ == '__main__':
    write_test_image()
