#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/20170606visualization.jpg "Visualization"
[image2_11]: ./examples/20170531before_grayscale.jpg "Grayscaling"
[image2_12]: ./examples/20170517_2before_grayscale.jpg "Grayscaling"
[image2_21]: ./examples/20170601after_grayscale.jpg "Grayscaling"
[image2_22]: ./examples/20170601_2after_grayscale.jpg "Grayscaling"
[image3_1]: ./examples/take001lr001e10va866.jpg "First Run Result Data"
[image3_2]: ./examples/20170516ep10lr001loss_plot.jpg "First Run Result Plot"
[image4_1]: ./examples/20170516ep15lr001drp1x5@full1loss_plot.jpg "First Dropout Result"
[image4_2]: ./examples/20170516ep10lr001drp2x5x2loss_plot.jpg "Second Dropout Result"
[image5_1]: ./examples/20170517ep20lr001drp2x5rgbNloss_plot.jpg "Normalization after 20 epochs"
[image5_2]: ./examples/20170607ep10lr001drp2x5rgbNloss_plot.jpg "Normalization after 10 epochs"
[image6]: ./examples/speed_50_001.jpg "Traffic Sign 1"
[image7]: ./examples/speed_80_001.jpg "Traffic Sign 2"
[image8]: ./examples/pedxing_001.jpg "Traffic Sign 3"
[image9]: ./examples/stop_001.jpg "Traffic Sign 4"
[image10]: ./examples/slippery_road_001.jpg "Traffic Sign 5"
[image11]: ./examples/20170607top_k.jpg "Softmax Probabilities"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.
Files Submitted
[iPython Notebook](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)
[iPython Notebook HTML version](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)
[Write-up](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/writeup.md)

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Dataset Visualization][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the images to make sure the ordering of the data does not affect the training of the network.

I have tried on converting images and not converting images to grayscale and sticking to colored images, and have found the results not having any significant difference. And so, I have deferred on using grayscale since some data are lost when converting to grayscale.

Here is an example of a traffic sign image results before and after grayscaling.

![Before grayscaling][image2_11]
![Before grayscaling][image2_12]
![After grayscaling][image2_21]
![After grayscaling][image2_22]

As a last step, I normalized the image data because it helps process the data faster.

I decided to defer to generate additional data because the result of the validation accuracy was already 93.1%.

To add more data to the the data set, I used the following techniques because ...

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flattening			| 5x5x16 image, outputs 400						|
| Fully connected		| 400 array, outputs 120						|
| RELU					|												|
| Dropout  		      	| keep_prob 0.5 								|
| Fully connected		| 120 array, outputs 84							|
| RELU					|												|
| Dropout  		      	| keep_prob 0.5									|
| Fully connected		| 84 array, outputs 43							|
|						|												|
|						|												|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
	- the softmax_cross_entropy_with_logits function to find cross_entropy with the one_hot_y and the logits as its parameters.
	- the AdamOptimizer and the Stochastic Gradient Descent but settled with the AdamOptimizer as optimizer as it is ideal with this amount of dataset.
	- 0.001 as initial learning rate after several trials with different learning rates.
	- 128 as batch size as it is ideal after several trials as well.
	- 10 as the ideal number of epochs after playing with different numbers as well.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


With no pre-processing and only the basic LeNet architecture, the highest validation accuracy reached by my model is 86.6% after tweaking with the learning rate, epochs and batch size values.

![First Run][image3_1]
![First Run][image3_2]

The resulting graph shows a significant overfitting; hence, I thought of adding a dropout layer after the last RELU layer of the second fully connected activation layer. The training accuracy reached 95% and the validation accuracy reached 87.1% after 10 epochs.

![Adding First Dropout][image4_1]

The overfitting was slightly resolved and so I added another one, this time after the RELU layer of the first fully connected activation layer. The training accuracy reached only 89.1% and the validation accuracy reached only 84% but the loss plot gave a better, much tighter curve.

![Adding Second Dropout][image4_2]

I then worked on the normalization of the images for pre-processing. I tried both the normalization of the grayscaled version of the images (see above) and the colored version. I dropped the grayscaling since there was no significant difference in the result with normalized colored images. The training accuracy reached 99.1% and the validation accuracy reached 94.8% after 20 epochs.

![Normalization after 20 Epochs][image5_1]

My final model results were:
* training set accuracy of 98%.
* validation set accuracy of 93%.
* test set accuracy of 92.7%.

![Normalization after 10 Epochs][image5_2]



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

The first image looks like it might be fairly easy to classify because it is in the list of sign classes with a fair amount of image samples but somehow the model gets it wrong. It may be the slight rotation of the image to the right.

![alt text][image7]

The second image maybe difficult to classify because it is rotated -90 degrees and likely much similar to another sign.

![alt text][image8]

The third image maybe difficult to classify because there is only a few number of samples in the training data.

![alt text][image9]

The fourth image should be easy to classify because it is very similar to the images in the training samples with a significant amount of sample images and true enough, the model gets it right.

![alt text][image10]

The fifth image should be difficult to classify because of the few number of sample images in the training data but for some reason, the model gets it right.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 50 km/h	      		| 30 km/h					 				|
| 80 km/h	      		| 120 km/h					 				|
| Pedestrians			| 30 km/h										|
| Stop Sign      		| Stop sign   									|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy on the test set of the model is 92.7% which is more than twice the model's guess accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .505         			| 50 km/h   									|
| .202    				| 80 km/h 										|
| .992					| Pedestrians									|
| .999	      			| Stop signs					 				|
| .925				    | Slippery Road      							|

![Softmax Probabilities for the 5 Web Images][image11]


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


