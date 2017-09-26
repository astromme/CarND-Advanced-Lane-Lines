import numpy as np
import cv2

# x, y
sourceBottomLeft = [264, 680]
sourceBottomRight = [1043, 680]
sourceTopLeft = [569, 470]
sourceTopRight = [718, 470]
source = np.array([sourceBottomLeft, sourceBottomRight, sourceTopLeft, sourceTopRight], np.float32)

destinationBottomLeft = [265, 720]
destinationBottomRight = [1045, 720]
destinationTopLeft = [265, 0]
destinationTopRight = [1045, 0]
destination = np.array([destinationBottomLeft, destinationBottomRight, destinationTopLeft, destinationTopRight], np.float32)

M = cv2.getPerspectiveTransform(source, destination)

def birdseye(image):
    """
    Apply a perspective transform to rectify binary image ("birds-eye view").
    """
    rows, cols = image.shape[0:2]
    return cv2.warpPerspective(image, M, (cols, rows))
