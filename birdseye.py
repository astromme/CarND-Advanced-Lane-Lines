import numpy as np
import cv2

# note: this could be more accurate by doing it on an undistorted image, not a distorted image
# x, y
sourceBottomLeft = [264, 680]
sourceBottomRight = [1043, 680]
sourceTopLeft = [590, 455]
sourceTopRight = [692, 455]
source = np.array([sourceBottomLeft, sourceBottomRight, sourceTopLeft, sourceTopRight], np.float32)

destinationBottomLeft = [265, 680]
destinationBottomRight = [1045, 680]
destinationTopLeft = [265, 100]
destinationTopRight = [1045, 100]
destination = np.array([destinationBottomLeft, destinationBottomRight, destinationTopLeft, destinationTopRight], np.float32)

M = cv2.getPerspectiveTransform(source, destination)

def birdseye(image):
    rows, cols = image.shape[0:2]
    return cv2.warpPerspective(image, M, (cols, rows))
