#**Project 3:  Use Deep Learning to Clone Driving Behavior** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* drive.py for driving the car in autonomous mode
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* PreProcess.py containing data preprocessing steps
* evaluate_model.ipynb for my work evaluating model
* README.md
 

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The PreProcess.py file contains accompanying pre-processing steps that were taken to make the model training more efficient.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

The model architecture is fully defined in model.py in the function define_conv_model (lines 35-102).  It consists of a convolution neural network inspired by the ResNet architecture.  It uses 5x5 filter sizes and depths between 6 and 16

The model includes RELU layers to introduce nonlinearity (part of define_conv_model) and the data is normalized in the model (see PreProcess.py function preprocess_batch_img) by dividing by 255

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (see model architecture in modely.py function define_conv_model)

The model was trained and validated on different data sets to ensure that the model was not overfitting (see PreProcess.py line 106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  I drove until I got about 20,000 rows of driver logs

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce dimensions of the image to a minimum then apply a convolutional neural network.

My first step was to use a convolution neural network model similar to ResNet. I thought this model might be appropriate because it worked well for traffic sign detection and is simple yet effective. I believe in parsimony

In order to gauge how well the model was working, I split my image and steering angle data into a training,validation, and test sets. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there was early stopping based on validation set and made sure the dropout layers were tuned to a number I felt comfortable with (ended up with 0.20).

The final step was to run the simulator to see how well the car was driving around track one. In early models, the car was very hesistant and did just drove straight, and thus off the tracks during curves.  To improve the driving behavior in these cases, I removed all datapoints where the steering angle is 0, which was the dominant steering angle. This forced the model to learn the non-zero steering angles and thus made the car more aggressive.  To futher encourge non-zero steering angles, I weighted the data by sqrt(abs(steering_angle)).  This means larger steering angles get more weight. This makes the car weave within the lane a bit but it's better than being too hesitant to steer. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture
The final model architecture (model.py function define_conv_model (lines 35-102)) consisted of a convolution neural network with the following layers and layer sizes:


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a few laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when veering off road.  These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]

I did not train on track 2

I did not need to augment the data set by creating other images although I heard that I could have used the right and left camera angles and modified the steering angles to get more data.

After the collection process, I had about 20,000 number of data points. I then preprocessed this data by downsampling the images by a factor of 8 and dividing pixel intensities by 255


I finally randomly shuffled the data set and put 80% of the data into train, 10% into validation, and 10% into test.  The test set wasn't really required but I kept it anyways.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  The test set was not used. The ideal number of epochs was 40 as evidenced by the early stopping criteria being met.  I used an adam optimizer so that manually training the learning rate wasn't necessar
