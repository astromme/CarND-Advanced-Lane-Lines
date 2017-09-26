import cv2
import numpy as np

def writeOverlayText(image, lines, frame, extraTextLines=[]):
    textLines = [
        'curvature: {0.left.radius_of_curvature:0.0f}m, {0.right.radius_of_curvature:0.0f}m'.format(lines),
        'offset from center: {:0.0f}cm'.format(lines.offset*100),
        'method: {0.method}'.format(lines),
    ] + extraTextLines
    if frame is not None:
        textLines.append('frame: {}'.format(frame))

    for i, textLine in enumerate(textLines):
        cv2.putText(
            img=image,
            text=textLine,
            org=(10,30*(i+1)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(255,255,255),
            thickness=2,
            lineType=cv2.LINE_AA)

def drawOverlay(image, birdseye_image, lines, M):
    # Create an image to draw the lines on
    birdseye_zero = np.zeros_like(birdseye_image).astype(np.uint8)
    color_birdseye = np.dstack((birdseye_zero, birdseye_zero, birdseye_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lines.left.fitx, lines.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lines.right.fitx, lines.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_birdseye, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    overlay = cv2.warpPerspective(color_birdseye, np.linalg.inv(M), (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, overlay, 0.3, 0)

    return result
