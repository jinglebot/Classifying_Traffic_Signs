# Traffic Sign Recognition

## Writeup Template

**Build a Traffic Sign Recognition Project**
---
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
### Files Submitted
[iPython Notebook](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)
[iPython Notebook HTML version](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)
[Writeup](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/writeup.md)

---
### Writeup / README

#### 1. Here is a link to my [project code](https://github.com/jinglebot/Classifying_Traffic_Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. I used the python library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the sign classes.

![Dataset Visualization][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data.

As a first step, I decided to shuffle the images to make sure the ordering of the data does not affect the training of the network.

I have tried on converting images to grayscale and sticking to colored images because I have found the results not having any significant difference. And so, I have deferred on using grayscale and used colored images upon proceeding to normalization to avoid data loss when converting to grayscale.

Here is an example of a traffic sign image results before and after grayscaling.

![Before grayscaling][image2_11]
![Before grayscaling][image2_12]
![After grayscaling][image2_21]
![After grayscaling][image2_22]

As a last step, I normalized the image data because it helps process the data faster.

I decided to defer to generate additional data because the result of the validation accuracy was already 93.1%.

#### 2. Final model architecture.

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


#### 3. How I trained your model.

To train the model, I used
	- the softmax_cross_entropy_with_logits function to find cross_entropy with the one_hot_y and the logits as its parameters.
	- the AdamOptimizer and the Stochastic Gradient Descent but settled with the AdamOptimizer as optimizer as it is ideal with this amount of dataset.
	- 0.001 as initial learning rate after several trials with different learning rates.
	- 128 as batch size as it is ideal after several trials as well.
	- 10, 15 and 20 epochs but settled at 10 as the ideal number of epochs after playing with the different numbers.

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

As a model architecture for predicting numbers with MNIST dataset, the LeNet architecture was chosen because it has already learned how to read figures. A few tweaks to customize it with the traffic sign dataset will save time for programmers from starting from scratch. These tweaks involve considering the differences on the dataset to be used, from grayscaled numbers to colored raster images, from a few thousand images to ten times the number or even more if augmentation is employed. In my model, the use of normalization, maxpool, dropout and additional channels for colored images are employed.

Initially, with no pre-processing and only the basic LeNet architecture, the highest validation accuracy reached by my model is 86.6% after tweaking with the learning rate, epochs and batch size values.

![First Run][image3_1]
![First Run][image3_2]

The resulting graph shows a significant overfitting; hence, I thought of adding a dropout layer right after the last RELU layer of the second fully connected activation layer. The training accuracy reached 95% and the validation accuracy reached 87.1% after 10 epochs.

![Adding First Dropout][image4_1]

Since the overfitting was slightly resolved, I added another one, this time after the RELU layer of the first fully connected activation layer. The training accuracy reached only 89.1% and the validation accuracy reached only 84% but the loss plot gave a better, much tighter curve.

![Adding Second Dropout][image4_2]

I then worked on the normalization of the images for pre-processing. I tried both the normalization of the grayscaled version of the images (see Pre-processing above) and the colored version. I dropped the grayscaling since there was no significant difference in the result with normalized colored images. The training accuracy reached 99.1% and the validation accuracy reached 94.8% after 20 epochs.

![Normalization after 20 Epochs][image5_1]

My final model results were:
* training set accuracy of 98%.
* validation set accuracy of 93%.
* test set accuracy of 92.7%.

![Normalization after 10 Epochs][image5_2]

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

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


#### 2. Model's predictions on the new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 50 km/h	      		| 30 km/h					 				|
| 80 km/h	      		| 120 km/h					 				|
| Pedestrians			| 30 km/h										|
| Stop Sign      		| Stop sign   									|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy on the test set of the model is 92.7% which is more than twice the model's guess accuracy.

#### 3. Certainty of the model when predicting on each of the five new images.

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .505         			| 50 km/h   									|
| .202    				| 80 km/h 										|
| .992					| Pedestrians									|
| .999	      			| Stop signs					 				|
| .925				    | Slippery Road      							|

![Softmax Probabilities for the 5 Web Images][image11]
