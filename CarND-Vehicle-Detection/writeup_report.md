## Vehicle Detection Project
### Wenzhe(Emma) Ding
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/window_size0.png
[image5]: ./output_images/window_size1.png
[image6]: ./output_images/window_size2.png
[image7]: ./output_images/window_size3.png
[image8]: ./output_images/scale.png
[image9]: ./output_images/initial_detection.png
[image10]: ./output_images/bboxes_and_heat.png
[image11]: ./output_images/output_bboxes.png
[image12]: ./output_images/comparison_hist_frames.png
[video1]: ./project_video_detection.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the [IPython notebook](SDC_nanodegree/CarND-Vehicle-Detection/CarND-Vehicle-Detection.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including `color_spaces`, `hog_channel`, `orientations`, `pixels_per_cell`, `cell_per_block`, `spatial_size` and `hist_bins`. Based on the result on course quizzes, using all hog channels always has better performance than using a perticular channel, so I set hog_channel = "ALL". The number of feature vector is 8460. The final value of `color_space` is determined by the accuracy of the classifier below.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC() in [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). The following table shows that the test accuracy of LinearSVC using different parameters. We chose 'YCrCb' as the 'color_space' due to its highest test accuracy.

| color_space         		|     hog_channel	        					| Test Accuracy |
|:---------------------:|:-------------------------------:| :-----------------------------------:| 
| RGB         		| ALL   							| 0.9778|
| HSV    	| ALL 	|0.9885|
| LUV					|	ALL											|0.9859|
| HLS	      	| ALL				|0.989|
| YUV    |  ALL			|0.989|
| YCrCb    |  ALL			|0.9913|

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

From course quizzes, I learned that restricting search area on the image could effectively filter out false positives (cells that are not cars but identified as cars). The following image shows the restricted search area of a full image.

![alt text][image3]

To select the right window size and overlap ratios for both x and y axis , I used the following possible values for window size: `(32,32), (64,64), (96,96), (128,128)` and the following ratios for xy_overlap: `(0, 0), (0.3, 0.3), (0.5, 0.5)`. The following graph shows `(32,32)` windows with `(0, 0),(0.3, 0.3),(0.5, 0.5)` overlapped ratio repectively.

![alt text][image4]

The following graph shows `(64,64)` windows with `(0, 0),(0.3, 0.3),(0.5, 0.5)` overlapped ratio repectively.

![alt text][image5]

The following graph shows `(96,96)` windows with `(0, 0),(0.3, 0.3),(0.5, 0.5)` overlapped ratio repectively.

![alt text][image6]


The following graph shows `(128,128)` windows with `(0, 0),(0.3, 0.3),(0.5, 0.5)` overlapped ratio repectively.

![alt text][image7]

It turned out that the combination of window size `(96,96)` and 0.5 overlapped ratio has the best detection results. 

I also compared the results of three different scales, namely 1, 1.5 and 2 in the function `find_cars`.

![alt text][image8]

Detection with scale 1.5 has the best results.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 1.5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are an intial detection image using function `find_cars`:

![alt text][image9]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video

Here's a [link to my video result](./project_video_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

### Here is one frame and its corresponding heatmap:

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image11]

In order to filter out false positives for a series of images in a video as well as refine the sizes of detected boxes, I implemented a method to keep detections that are found at or near the same position in several previous frames.
```
if len(history) > 0:
      his_heat = np.zeros_like(img[:,:,0]).astype(np.float)
      for hm in history:
          his_heat += hm
      his_heat = apply_threshold(his_heat,len(history)-1)
      heat += his_heat
```

### Here is six frames and its corresponding heatmaps with each one considering previouse 5 frames:

![alt text][image12]

It is clear that in the last frame, the size of the boxes is more accuracy compared with that without considering historical frames.

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried to use [Grid Search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to train the classifier which is used to differentiate cars vs. non-cars. However, it takes a very long time to tune the parameters, such as `C` and `kernel`. Therefore, I choose LinearSVC for its low computational cost. Also, In some frames of the video, we see that the detected boxes are not large enough to mark a whole car. My pipeline is likely to fail when encountering situations such that driving at night when it's hard to detect vehicle vs. non-vehicles. 

