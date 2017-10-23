
---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/0camera_caliberation.png 
[image2]: ./output_images/1undistort.png 
[image3]: ./output_images/2sample.png 
[image4]: ./output_images/3sample_undistored.png 
[image5]: ./output_images/4channel.png 
[image6]: ./output_images/5channel_thresh.png 
[image7]: ./output_images/6gradient.png 
[image8]: ./output_images/7thresh_magnitude.png 
[image9]: ./output_images/8thresh_dir.png 
[image10]: ./output_images/9pipe_res.png 
[image11]: ./output_images/10undist.png 
[image12]: ./output_images/11color_pipeline.png 
[image13]: ./output_images/12histogram.png 
[image14]: ./output_images/13rec_fit.png 
[image15]: ./output_images/14fitpoly.png 
[image16]: ./output_images/15sample_output.png
[image17]: ./output_images/16all_output.png 
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

The image after undistortion is shown as below:

![alt text][image4]

It is clear the shape of the car hood has changed after undistortion.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of apply thresholding on some color spaces. 

![alt text][image6]

It turned out HLS-S channel has the best detection of lane lines.

Here is my output for producing thresholded binary images. 

![alt text][image10]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 21st code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = undist_sample.shape[:2]
src = np.float32([(610,450),
                  (710,450), 
                  (200,700), 
                  (1100,700)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image11]

The following images demonstarte all the test images after applying color thresholding and perspective transform.

![alt text][image12]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I looked at the histogram of the sample image and used both `sliding_window_polyfit` (code cell 28) and `polyfit_using_prev_fit` (code cell 32) to fit my lane lines with a 2nd order polynomial. The following images show the result from `sliding_window_polyfit` and `polyfit_using_prev_fit` respectively.

![alt text][image14]

![alt text][image15]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 36 in my code in function `calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds)`. After calculating the radius of curvature of the land as well as the position of the vehicle, I used function `put_text` to write these values on the images.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 39 in function `(original_img, binary_img, l_fit, r_fit, Minv)`.  Here is an example of my result on a test image:

![alt text][image16]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started with some easy samples and the results of my line detection pipeline worked well untill the vehicle entered a shaded area. Then I worked with the most changllenged image - the one with most shades - in the test images to get a feel of selecting color space and tuning the parameters. I revised my pipeline based on the results of the most changllenged image and it also worked for other images. Some of the detected lane lines are wobbly but the output has an overall precise detection.  My pipeline is likely fail when there are white or light color cars or white signs in front of the vehicle, because then it may consider those kinds of objects as part of the lanes. To make it more robust, possible ways include selecting certain areas in the perspective transformed image to apply color thresholding rather than the whole image.  
