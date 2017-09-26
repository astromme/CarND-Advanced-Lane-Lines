import numpy as np

class FitFailedError(Exception):
    pass


class Line():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    max_curvature_change = 0.2

    margin = 100

    def __init__(self, ploty, logger=None):
        self.logger = logger
        self.ploty = ploty

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = []

        #y values for detected line pixels
        self.ally = []

    @property
    def fitx(self):
        return self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2]

    def add_fit(self, fit, xpixels, ypixels):
        n = 10

        self.allx.append(xpixels)
        self.ally.append(ypixels)

        assert len(xpixels) == len(ypixels)

        if len(self.allx) > n:
            self.allx.pop(0)
            self.ally.pop(0)

        self.current_fit = fit
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (1 - 1/n) * self.best_fit + (1/n) * fit


    @property
    def radius_of_curvature(self):
        x = np.concatenate(self.allx).ravel()
        y = np.concatenate(self.ally).ravel()
        return self._calculate_curvature(x, y)


    """and vehicle position with respect to center"""

    def _calculate_curvature(self, x, y):
        """
        Determine the curvature in meters of the lane.
        """
        y_eval = np.max(y)

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y*self.ym_per_pix, x*self.xm_per_pix, 2)

        # Calculate the radius of curvature
        return ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    def find_with_sliding_window(self, nonzerox, nonzeroy, x_base, nwindows=9, max_height=720, minpix=50):
        # Set height of windows
        window_height = np.int(max_height/nwindows)

        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = max_height - (window+1)*window_height
            win_y_high = max_height - window*window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin

            # if self.logger and logger.enabled:
            #     # Draw the windows on the visualization image
            #     cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),
            #     (0,255,0), 2)


            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        return x, y

    def find_with_margin_around_previous(self, nonzerox, nonzeroy):
        if self.current_fit is None:
            raise FitFailedError('no prevous fit')

        lane_inds = ((nonzerox > (self.current_fit[0]*(nonzeroy**2) + self.current_fit[1]*nonzeroy +
        self.current_fit[2] - self.margin)) & (nonzerox < (self.current_fit[0]*(nonzeroy**2) +
        self.current_fit[1]*nonzeroy + self.current_fit[2] + self.margin)))

        # extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if len(y) == 0:
            raise FitFailedError('no points detected')

        abs_curvature = abs(self._calculate_curvature(x, y))
        if np.sqrt(abs(abs_curvature - self.radius_of_curvature)) / abs_curvature > self.max_curvature_change:
            raise FitFailedError('curvature changed more than {:0.0f}%'.format(self.max_curvature_change*100))

        return x, y
