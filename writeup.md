
# Udacity Self-Driving Car Nanodegree
---

## Project #3: Behavioral Cloning
---

The goals of this project are to:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track without leaving the road
* Summarize the results with a written report (this file)
---

[//]: # (Image References)

[image1]: ./examples/center_2017_07_20_16_43_38_565.jpg "Normal Image"
[image2]: ./examples/center_2017_07_20_16_43_38_565_flipped.jpg "Flipped Image"
[image3]: ./examples/train-valid-split.png "Flipped Image"

### Writeup
#### 1. Provide a writeup that includes all of the [rubric points](https://review.udacity.com/#!/rubrics/432/view) and describes how I addressed each point in my implementation.
You're reading it!

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model (model.py lines 110-134) is a convolution neural network consisting of two convolutional layers followed by two dense layers separated by a dropout layer. ReLU activations are used to introduce nonlinearity. 

Additionally, cropping and normalizing layers are used to preprocess the input images.

#### 2. Attempts to reduce overfitting in the model

The model contains a 50% dropout layer in order to reduce overfitting (model.py line 127). 

Extra training data was also created to reduce the likelihood of overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 192-196). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I tuned the left/right camera angle bias (model.py line 16) by training the model, observing it perform, and increasing this parameter until the model felt sufficiently 'apprehensive' of the edge of the road. Additionally, since the model consistently under-steered turns, I added a small squared term to all angles so that the model which immediately made the model more responsive around turns.

Due to time constraints, I trained the model on as few epochs as would produce a satifactory model. Two epochs seemed to be sufficient, taking approximately 5 minutes per epoch to train on my laptop.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 191).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I trained strictly on center lane driving, however I used the left and right camera images to simulate recovery.

For details about how I created the training data, see point 3 in the next section. 

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start small and incrementally add complexity as appropriate.

My first step was to use a convolution neural network model similar to a small LeNet architecture. I thought this model might be appropriate because even the simplest network with no activations produced mediocre results (i.e. it made it around the first turn).

The next step was to augment the data with random horizontal flipping to reduce a left-turn bias in the training data. I further stretched the data by adding the left and right camera images with a small bias added or subtracted from the steering angle.

The final step was to run the simulator to see how well the car was driving around track 1. The model consistently under-steered around large turns. To improve the driving behavior in these cases, I added a small squared term to the angle measurements (effectively instructing the model to be less timid, i.e. "if you think you should turn...turn more!"). 

After retraining with this extra term, the model produced sufficiently sharp turns and was able to drive the vehicle autonomously around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 110-134) consists of a convolution neural network with one 5x5 filter followed by one 3x3 filter with a depth of 10 and 20, respectively. This is followed by two dense layers (256 and 1) separated by a 50% dropout layer.

The model includes RELU activations after each layer except the last to introduce nonlinearity. Additionally two preprocessing steps are defined - the first is a Cropping2D layer which removes the top 50 pixels and the bottom 20 pixels, the second is a Lambda layer for normalizing the pixel values from [0,255] to [-0.5,0.5].

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded five laps on track 1 using center lane driving. Three laps were driven in the fowards direction, and the other 2 laps were driven in the reverse direction. I repeated this process on track 2 in order to get more data points and to help the model generalize better.

Although the forum advice suggests that mouse or joystick data produces data of higher quality than the keyboard, I only had a keyboard so I tried to combat the lower quality data by collecting and training on a greater quantity of data.

Here is an example image of center lane driving:

![alt text][image1]

I randomly flipped images and their angles in an attempt to reduce the bias for left turns. Here is the flipped version of the previous image:

![alt text][image2]

After the collection process, I had 37302 data points (including left and right camera angles) and put 25% of the data into a validation set. The validation set was created by taking a consectutive slice out of each directory (model.py lines 152-179). Since each directory stores the images for one lap, we end up with one validation slice per lap. Each validation slice starts at a random position in the recording, as illustrated by the graphic below:

![alt text][image3]

Training and validation preprocessing was done in the batch generator (model.py lines 14-70) and consisted of random horizontal flipping and random shuffling.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the validation loss approximately equal to the training loss and the model successfully driving around the track.
