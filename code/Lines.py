import numpy as np
from Line import Line, FitFailedError
import matplotlib.pyplot as plt
import cv2

class Lines:
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    def __init__(self, height, logger=None):
        self.logger = logger
        self.ploty = np.linspace(0, height-1, height)
        self.left = Line(self.ploty, logger)
        self.right = Line(self.ploty, logger)
        self.method = None
        self.detected = False
        self.offset = None

    def add_candidate_fit(self, leftx, lefty, rightx, righty, method):
        if len(lefty) == 0 or len(righty) == 0:
            print('points are empty')
            raise FitFailedError('points are empty')

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


        if np.array_equal(left_fit, right_fit):
            raise FitFailedError('lines are equal')

        bottomLeft = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
        bottomRight = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]

        lane_width = bottomRight - bottomLeft
        midpoint = bottomLeft + lane_width / 2

        center_of_image = 1280/2

        self.offset = (center_of_image - midpoint)*self.xm_per_pix
        self.detected = True
        self.method = method
        self.left.add_fit(left_fit, leftx, lefty)
        self.right.add_fit(right_fit, rightx, righty)

    def find(self, birdseye_image):
        image_sum = np.sum(birdseye_image)
        nonzero = birdseye_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        try:
            leftx, lefty = self.left.find_with_margin_around_previous(nonzerox, nonzeroy)
            rightx, righty = self.right.find_with_margin_around_previous(nonzerox, nonzeroy)
            self.add_candidate_fit(leftx, lefty, rightx, righty, 'margin_around_previous')
            self.log_fit(leftx, lefty, rightx, righty, birdseye_image, 'margin_around_previous')

        except (FitFailedError) as e:
            try:
                histogram = np.sum(birdseye_image[birdseye_image.shape[0]//2:,:], axis=0)
                plt.plot(histogram)
                self.logger.plot('histogram', plt)
                plt.clf()

                midpoint = np.int(histogram.shape[0]/2)
                leftx_base = np.argmax(histogram[:midpoint])
                rightx_base = np.argmax(histogram[midpoint:]) + midpoint

                leftx, lefty = self.left.find_with_sliding_window(nonzerox, nonzeroy, leftx_base)
                rightx, righty = self.right.find_with_sliding_window(nonzerox, nonzeroy, rightx_base)
                self.add_candidate_fit(leftx, lefty, rightx, righty, 'sliding_window: {}'.format(str(e)))
                self.log_fit(leftx, lefty, rightx, righty, birdseye_image, 'sliding_window')

            except (FitFailedError) as e:
                self.detected = False
                self.method = 'failed_to_detect: {}'.format(str(e))

        if image_sum < 3000:
            self.detected = False
            self.method = 'failed_to_detect: too little data'

    def log_fit(self, leftx, lefty, rightx, righty, birdseye_image, fit_type):
        if not self.logger or not self.logger.enabled:
            return

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((birdseye_image, birdseye_image, birdseye_image))*255
        # Color in left and right line pixels

        for ly, lx, ry, rx in zip(lefty, leftx, righty, rightx):
            out_img.itemset((ly, lx, 0), 255)
            out_img.itemset((ry, rx, 2), 255)

        if fit_type == 'sliding_window':
            window_img = np.zeros_like(out_img)

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self.left.fitx-self.left.margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left.fitx+self.left.margin,
                                          self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self.right.fitx-self.right.margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right.fitx+self.right.margin,
                                          self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        elif fit_type == 'margin_around_previous':
            result = out_img

        plt.imshow(out_img)
        plt.plot(self.left.fitx, self.ploty, color='yellow')
        plt.plot(self.right.fitx, self.ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        self.logger.plot(fit_type, plt)
        plt.clf()
