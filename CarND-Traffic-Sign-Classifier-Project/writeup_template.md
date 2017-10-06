# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/original.png "Original"
[image3]: ./examples/grayscale.png "Grayscaling"
[image4]: ./examples/0_download.png "Traffic Sign 1"
[image5]: ./examples/13_download.png "Traffic Sign 2"
[image6]: ./examples/18_download.png "Traffic Sign 3"
[image7]: ./examples/3_download.png "Traffic Sign 4"
[image8]: ./examples/7_download.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wzding/SDC_nanodegree/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. The data is downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in with resized images of 32x32. I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a line plot showing how the number of samples per label in train and validation data set. The proportion of labels in the train and validation data have similar patterns. But the number of samples per label does not seem to be balanced in train and validation set. For example, the ratio between most frequently labels and least frequently labels is 8 of validation set while the ratio is around 11 in train data. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because it changes the range of pixel intensity values from 0 to 255 to -1 to 1. Then I fit these normalized data sets into model training and obtained the results of the model.

As a second step, I decided to convert the images to grayscale because my model may learn better from compressed images. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

 

To add more data to the the data set, I used the following techniques because a balanced data set often improve model accuracy. For each labels with samples less than 2010/8, I random sampled data from training data and append them to exsiting training data.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    |  1x1 stride,  outputs 10x10x6			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten				| inputs 5x5x16,  outputs 400      									|
| Fully connected		|inputs 400,  outputs 120      									|
| RELU					|												|
| Fully connected		|inputs 120,  outputs 84      									|
| RELU					|												|
| Fully connected		|inputs 84,  outputs 43      									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 128 batch size, 20 epoches, and learning rate 0.001. My optimizaer is AdamOptimizer because it makes learning less sensitive to hyperparameters.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.930
* test set accuracy of 0.902

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because it is hard to differentiate the image from its background color.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)     		| Speed limit (120km/h)   									| 
| Yield					| Yield											|
| General caution     			|End of no passing								|
| Speed limit (60km/h)      		|Speed limit (60km/h)					 				|
| Speed limit (100km/h)			| Speed limit (100km/h)     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is much less than the accuracy on the test set of 0.902.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 39th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit (probability of 0.67), but it turned out that is is a different speed limit - 20km/h rather than 120km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Stop sign   									| 
| .17     				| U-turn 										|
| .06					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image, the model is relatively sure that this is a speed limit (probability of 0.67), but it turned out that is is a different speed limit - 20km/h rather than 120km/h. 
For the third image, the model is relatively sure that this is a speed limit (probability of 0.67), but it turned out that is is a different speed limit - 20km/h rather than 120km/h. 
For the forth image, the model is relatively sure that this is a speed limit (probability of 0.67), but it turned out that is is a different speed limit - 20km/h rather than 120km/h. 
For the fifth image, the model is relatively sure that this is a speed limit (probability of 0.67), but it turned out that is is a different speed limit - 20km/h rather than 120km/h. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


