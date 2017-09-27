# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied Gaussian smoothing on the grayscale image.

<img src="./examples/grayscale.jpg" alt="Grayscale" width="400">

Afterwards, I applied Canny edge detector on this image and got the image below:

<img src="./examples/canny_edge.jpg" alt="Cannyedge" width="400">

The following image shows what it looks like after I used a mask on the image.

<img src="./examples/marked_edge.jpg" alt="Markededge" width="400">

Then I ran Hough on edge detected image.

<img src="./examples/hough.jpg" alt="Markededge" width="400">

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by introducing two functions: average_slope_intercept and convert_line_points. The first function average_slope_intercept is used to get two single lines -one on the left and the other on the right - by averaging and extrapolating identified lines. Afterwards, I used function convert_line_points to convert each line represented in slope and intercept into pixel points. We then obtained two single lines of the Hough lines.

<img src="./examples/hough_transform.jpg" alt="Markededge" width="400">

Finally, I drew these two lines on the original image and obtained the final image as follows.

<img src="./examples/final.jpg" alt="final" width="400">

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
