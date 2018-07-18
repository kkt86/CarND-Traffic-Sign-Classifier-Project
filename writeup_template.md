# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goal of this project is to build classifier model (based on a convolutional neural 
network architecture), which is capagle to recognize and classify images of traffic signs.
  

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
  
---
### Writeup / README

In this README, we address the following poinds:

1. Dataset summary & exploration
2. Design and test a model architecture 
3. Test a model on new images 

### Data Set Summary & Exploration

Three separate datasets are provided, for training, validation and testing, containing
colored images of traffic signs, rescaled to 32x32 pixels.  

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

---

In the following figure, we can see the distributions of the distinct signs in 
the training, validation and testing datasets:

[image1]: ./plots/distributions_labels.png
![][image1]

We can also observe some samples of the different images, present in the training dataset:

[image2]: ./plots/sample_signs.png
![][image2]


### Design and Test a Model Architecture

In the preprocessing phase, different techniques have been applied:

1. Image Gaussian bluring
2. Conversion of the RGB to YUV channels
3. Histogram equilization of the Y-channel

Here is a visualization of the original image, the output of the two single preprocessing functions, and their final combination:

[image3]: ./plots/preprocessed_images.png
![][image3]

Furthermore, we augmented the training data by adding 5 more distored images, for each one, present in the original training data. The distorsion has been done by means of the following functions:

1. Random rotation of an angle between -30° and 30° degrees
2. Random scaling of the x and y axis of the original image

The image distorsion (as well as the results from the two single functions) can be visualized in the following image:

[image4]: ./plots/distorted_images.png
![][image4]

Finally, we provide a visualization of the original and final (preprocessed and distorted image):

[image5]: ./plots/final_preprocessed_image.png
![][image5]

---

The model architecture, used in this project, is the standard LeNet architecture, plus additional dropout layers:

| Layer	|	Description	| 
|:-------:|:--------------:| 
| Input   | 32x32x1 Y-channel image | 
| Convolution 5x5 | 5x5 stride, valid padding, outputs 28x28x6 |
| RELU	|						|
| Max pooling	 | 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5 | 5x5 stride, valid padding, outputs 10x10x16 |
| RELU	|						|
| Max pooling	 | 2x2 stride, same padding, outputs 5x5x16 | 
| Fully connected | Input: 400, Output: 120 |   
| RELU	|						|
| Dropout	| Keep probability: 0.7 |
| Fully connected | Input: 120, Output: 80 |   
| RELU	|						|
| Dropout	| Keep probability: 0.7 |  
| Fully connected | Input: 80, Output: 43 |   		
 For training the model, 150 epochs has been utilized, with learning 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

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
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


