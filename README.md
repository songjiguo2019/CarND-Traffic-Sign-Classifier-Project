# **Traffic Sign Recognition** 

## Note: all code is in P3.py
## Note: transformed_data_2.pickle is too big (>200MB) and can be generated if needed.

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

[image_dataset_hist]: ./outputimages/number_of_classes.png "number_of_classes"
[image_random_image]: ./outputimages/random_train_image.png "random_image"
[image_before_augmentation]: ./outputimages/orignal_image.png "before_aug_image"
[image_after_augmentation]: ./outputimages/transformed_image.png "before_aug_image"
[image_before_gray]: ./outputimages/random_train_image_preprocess.png "before_gray_image"
[image_after_gray]: ./outputimages/grayscaled_train_image_preprocess.png "after_gray_image"
[image_after_norm]: ./outputimages/normalized_train_image_preprocess.png "after_norm_image"
[image_dataset_hist_augmented]: ./outputimages/normalized_classes.png "augmented_classes"

[image_1x_large]: ./test_images/1x_large.png "1x"
[image_2x_large]: ./test_images/2x_large.png "2x"
[image_3x_large]: ./test_images/3x_large.png "3x"
[image_5x_large]: ./test_images/5x_large.png "5x"
[image_6x_large]: ./test_images/6x_large.png "6x"
[image_8x_large]: ./test_images/8x_large.png "8x"
[image_9x_large]: ./test_images/9x_large.png "9x"
[image_stop_large]: ./test_images/stop_large.png "stop"


[image_top_five]: ./outputimages/top_five.png "number_of_classes"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

As explained earlier, all code can be found in P3.py. Each stage will be explained in the next few subsections:

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Data set is orignally from http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset and we can import trainning 
data set, validation set and test set (*train.p, valid.p and test.p*). Here is the summary statistics of the traffic 
signs data set (computated by *init()* in P3.py):

* The size of training set is 86430
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of all data set (*train, valid and test*). It is a histogram chart showing the distribution of 
the training class. 


![alt text][image_dataset_hist]

As we can see, some classes are under-represented. We also tried to load a random image to see if we can correctly load the augmented traffic sign image. Here is one:

![alt text][image_random_image]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Data augmentation methods in (*transform_training_data()*) and (*transform_image()*) are used to add more images for 
the under-represented training classes. The methods I used include:
 - Rotate image within certain range of angles
 - Using Shear algorithm apply affine transform
 - Apply translations over the image (a random uniform distribution is used to generate different parameters)

Here are some results before applying above augmentation methods and after: 
![alt text][image_before_augmentation]
![alt text][image_after_augmentation]

Also I converted the images to grayscale (*pre_process()*) and applied the normalization function (*pre_process()*) 
for the grayscaled images. Here is an example of traning image before and after grayscaling/normalization:

![alt text][image_before_gray]
![alt text][image_after_gray]
![alt text][image_after_norm]

The difference between the original data set (top) and the augmented data set (bottom) is shown in the 
following histogram figures:
![alt text][image_dataset_hist]
![alt text][image_dataset_hist_augmented]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (see *LeNet()* in P3.py):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6, conv kernel:2x2 |
| Convolution 5x5       | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16, conv kernel:2x2  |
| Flatten   			| 1x400											|
| Fully connected layer 1 with dropout		| 1x12						|
| RELU					|												|
| dropout				|	    0.8										|
| Fully connected layer 1 with dropout		| 1x84						|
| RELU					|												|
| dropout				|		0.8										|
| Softmax ()			|       43        								|
|						|												|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following paramters:
 - BATCH_SIZE = 128
 - EPOCHS = 50
 - AdamOptimizer with a learning rate = 0.001
 
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.922
* test set accuracy of 0.915

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it
chosen?  The first architecture tried was the default LeNet without
dropout and without data augmentation. It obviously did not work very
well.
* What were some problems with the initial architecture?  The problem
was it ended up around less than 80% accuracy.
* How was the architecture adjusted and why was it adjusted? Typical
adjustments could include choosing a different model architecture,
adding layers (pooling, dropout, convolution, etc), using an RELU
activation function or changing the activation function. One common
justification for adjusting an architecture would be due to
overfitting or underfitting. A high accuracy on the training set but
low accuracy on the validation set indicates over fitting; a low
accuracy on both sets indicates under fitting. The model ended up
around 92.2% validation accuracy.  I used the BATCH_SIZE 128.  I also
increased the EPOCHS from 20 to 50. It helped quite a little bit on
the accuracy.

* What are some of the important design choices and why were they
chosen? For example, why might a convolution layer work well with this
problem? How might a dropout layer help with creating a successful
model?  The most important change is the data augmentation before
applying the LeNet. Also adding pooling layers and dropout helps.

If a well known architecture was chosen:
* What architecture was chosen?  Here I chose the LetNet-5 layer
architecture for the project. It has 2 convolution layers and 3 fully
connected layers
* Why did you believe it would be relevant to the traffic sign
application?  With more layers, it extracts more images features. With
several try-outs, it works pretty good on the validation and test
accuracy.
* How does the final model's accuracy on the training, validation and
test set provide evidence that the model is working well?  Data
augmentation, normalization helped better training the model. Dropout
helped avoiding the overfitting, thus helped the validation and
testing accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are eight German traffic signs that I found on the web: all the 8 images are screenshots of Road Traffic Signs in 
Germany (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The image qualities are very similar to the training datasets. It may be difficult to class for the high 
quality images because training datasets are not high quality images.


![alt text][image_stop_large]![alt text][image_1x_large] ![alt text][image_2x_large] ![alt text][image_3x_large]
 
![alt text][image_5x_large] ![alt text][image_6x_large]![alt text][image_8x_large]![alt text][image_9x_large]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		                    | Stop   									| 
| Right-of-way at the next intersection | Right-of-way at the next intersection 	|
| Speed limit (30km/h)					| Speed limit (30km/h)						|
| Priority road	      		            | Priority road	    		 				|
| Keep right			                | Keep right      							|
| General caution			            | General caution  							|
| Road work			                    | Road work      							|
| Turn left ahead			            | Turn left ahead 							|


The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. It shows that the model really 
works well at prediction.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in *prediction()* in P3.py.

For all eight images, the model is with a probability of 1.0 predition accuracy. 
Below is an example for the *Stop Sign* image prediction. The top five soft max probabilities for Stop Sign image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop         									| 
| 0.0     				| Yield 										|
| 0.0					| No Pass                                       |
| 0.0	      			| Priority road                                 |
| 0.0				    | Right Turn      							    |

As you can see, the model is always pretty sure that the first image is a stop sign (probability of 1), and the image 
does contain a stop sign. To see the top five soft max probabilities, here is the prediction result for all 8 test 
images:

![alt text][image_top_five]
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


