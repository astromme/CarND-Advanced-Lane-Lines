# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration1]: ./camera_cal/calibration1.jpg "Distorted Chessboard"
[calibration2]: ./examples/undistorted_calibration1.jpg "Undistorted Chessboard"
[stage0]: ./examples/image1-stage0-image.jpg "Distorted Image"
[stage1]: ./examples/image1-stage1-undistorted.jpg "Undistorted Image"
[stage2]: ./examples/image1-stage2-hue.jpg "Hue"
[stage3]: ./examples/image1-stage3-lightness.jpg "Lightness"
[stage4]: ./examples/image1-stage4-saturation.jpg "Saturation"
[stage5]: ./examples/image1-stage5-saturation+0.3lightness.jpg "Saturation + 0.3*Lightness"
[stage6]: ./examples/image1-stage6-gradx.jpg "Gradient x"
[stage7]: ./examples/image1-stage7-grady.jpg "Gradient y"
[stage8]: ./examples/image1-stage8-mag_binary.jpg "Gradient magnitude"
[stage9]: ./examples/image1-stage9-dir_binary.jpg "Gradient direction"
[stage10]: ./examples/image1-stage10-combined_binary.jpg "Combined binary gradient"
[stage11]: ./examples/image1-stage11-birdseye.jpg "Birdseye"
[stage12]: ./examples/image1-stage12-histogram.jpg "Histogram of bottom section of birdseye"
[stage13]: ./examples/image1-stage13-sliding_window.png "Sliding Window"
[stage14]: ./examples/image1-stage14-result.jpg "Output"
[stage15]: ./examples/image1-stage15-grid.jpg "Debug output (output, S+0.3L, Binary, Birdseye)"
[video1]: ./project_video_output.mp4 "Output Video"
[video2]: ./challenge_video_output.mp4 "Challenge Video"
[video3]: ./harder_challenge_video_output.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in `code/calibrate_camera.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][calibration1]
distorted

![alt text][calibration2]
undistorted

This code also saves the `mtx` and `dist` to `calibration.pickle` to speed up future runs of the code.

### Pipeline

#### 1. Provide an example of a distortion-corrected image.

I wrote code that dumped out the pipeline at various frames from the video. See this pipeline in the `pipeline()` function in `code/lane_lines.py` One example of a frame is as follows:
![alt text][stage0]

This first got undistorted
![undistorted][stage1].

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I used a combination of color and gradient thresholds to generate a binary image (thresholding is in `code/thresholds.py`). The next 9 images show each of these stages. See `pipeline()` from `code/lane_lines.py` to see the exact values for thresholds, kernel sizes, and order of operations.

I tried changing the sobel kernel size but it didn't seem to help much.

Hue:
![hue][stage2]

Lightness:
![lightness][stage3]

Saturation
![saturation][stage4]

Saturation+0.3*lightness (used as gradient inputs)
![hue][stage5]

Gradient x:
![gradx][stage6]

Gradient y:
![grady][stage7]

Gradient magnitude:
![mag][stage8]

Gradient direction:
![direction][stage9]

Gradient combined
![binary][stage10]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

See `code/birdseye.py` for my perspective transform code. I chose the hardcode the source and destination points in the following manner:

```python
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
```

I improved the birdseye image by taking coordinates from the undistorted image rather than the original distorted image.

pipeline stage 11: birdseye
![birdseye][stage11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

You can find my code for finding lane-line pixels and fitting polynomials in `code/Line.py` and `code/Lines.py`. The basic strategy is to use a sliding window with a histogram to first identify the lines, then use a margin around the previous frame's lines to speed up computation for future frames. When the lines don't make sense (e.g. they cross, or are the same, or change too much) then the code reverts back to the sliding window. If that fails, it keeps the previous frame's lines.

I also average the lines over the most recent 10 frames, this is in `Line.add_fit()`

![alt text][stage13]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Find my curvature code in `Line._calculate_curvature`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`overlay.py` contains my code to project the lines back onto the original image, and to add debug text like frame number which helped pinpoint problems. I also compose a grid of 4 images, including the output, birdseye, S+0.3L, and the combined binary threshold. This made debugging much easier.

![alt text][stage15]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]. You can also see the subpar performance on the [challenge1][video2] and [challenge2][video3].

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used a straightforward approach that was demonstrated in the lessons, which can be summarized as trying to isolate binary pixels as lines, then do a polyfit. This is coupled with a history so that if a given frame is bad the previous frame's lines can be used.

My pipeline will fail if the S+0.3L image doesn't capture the lane lines for more than a few frames, for example in a dark tunnel or when the lines are worn off. I could experiment with darker images and try using an adaptive brightness that looked at the total lightness in the image or the local lightness to better identify lines.

 It'll also fail if a line isn't visible, for example this happens in one of the tight turns of the harder challenge video. I could assume that the line simply moves over by as much as the other line moves. Lanes themselves don't change width that much.

It will also fail when another car is right in front of the camera. I could use a car classification CNN to recognize and remove the car from the image before thresholding.
