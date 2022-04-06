# Human Gender and Age Detection using OpenCV & Deep Learning

This model helps you find age & gender
of person in image

## Brief Working

In this project, We have used Deep Learning
to accurately identify the gender and age of
a person from a single face image.
The predicted gender may be one of ‘Male’
and ‘Female’, and the predicted age may
e one of the following ranges- (0 – 2),
(4 – 6), (8 – 12), (15 – 20), (25 – 32),
(38 – 43), (48 – 53), (60 – 100)
8 nodes in the final softmax layer).
It is very difficult to accurately
guess an exact age from a single
image because of factors like makeup,
lighting, obstructions, and facial
expressions. And so, We made this a
classification problem instead of making
it one of regression

# The CNN Architecture

We have used a very simple convolutional neural network architecture, similar to the CaffeNet and AlexNet. The network uses 3 convolutional layers, 2 fully connected layers and a final output layer

- Conv1 : The first convolutional layer has 96 nodes of kernel size 7

- Conv2 : The second conv layer has 256 nodes with kernel size 5

- Conv3 : The third conv layer has 384 nodes with kernel size 3

The two fully connected layers have 512 nodes each

# Dataset

We used the Adience [Dateset](https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification)

This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions :

- noise
- lighting
- pose
- appearance

The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license.

It has a total of 26,580 photos of 2,284 subjects in 8 age ranges (as mentioned above) and is about 1GB in size.\
 The models we used have been trained on this dataset

# Libraries Required

Libraries are of Python

```bash
  OpenCV
```

```bash
  argparse
```

## Install packages

- OpenCV

```bash
  pip install opencv-python
```

- argparse

```bash
    pip install argparse
```

# Contents of Project

Libraries are of Python

- opencv_face_detector.pbtx
- opencv_face_detector_uint8.pb
- age_deploy.prototxt
- age_net.caffemodel
- gender_deploy.prototxt
- gender_net.caffemodel
- tester picture
- detect.py

For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

1.  We use the argparse library to create an argument parser so we can get the image argument from the command prompt

2.  We make it parse the argument holding the path to the image to classify gender and age for

3.  For face, age, and gender, initialize protocol buffer and model

4.  Initialize the mean values for the model and the lists of age ranges and genders to classify from

5.  Now, use the readNet() method to load the networks. The first parameter holds trained weights and the second carries network configuration

6.  Let’s capture video stream in case you’d like to classify on a webcam’s stream. Set padding to 20

7.  Now until any key is pressed, we read the stream and store the content into the names hasFrame and frame. If it isn’t a video, it must wait, and so we call up waitKey() from cv2, then break

8.  Let’s make a call to the highlightFace() function with the faceNet and frame parameters, and what this returns, we will store in the names resultImg and faceBoxes. And if we got 0 faceBoxes, it means there was no face to detect.
    Here, net is faceNet- this model is the DNN Face Detector and holds only about 2.7MB on disk.

- Create a shallow copy of frame and get its height and width.
- Create a blob from the shallow copy.
- Set the input and make a forward pass to the network.
- faceBoxes is an empty list now. for each value in 0 to 127, define the confidence (between 0 and 1). Wherever we find the confidence greater than the confidence threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates and append a list of those to faceBoxes.
- Then, we put up rectangles on the image for each such list of coordinates and return two things: the shallow copy and the list of faceBoxes.

9.  But if there are indeed faceBoxes, for each of those, we define the face, create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values

10. We feed the input and give the network a forward pass to get the confidence of the two class. Whichever is higher, that is the gender of the person in the picture

11. Then, we do the same thing for age

12. We’ll add the gender and age texts to the resulting image and display it with imshow()

The code can be divided into four parts:

```bash
1. Detect Faces
```

```bash
2. Detect Gender
```

```bash
3. Detect Age
```

```bash
4. Display output
```

Let us have a look at the code for gender and age prediction using the DNN module in OpenCV

## Detect Face

We will use the DNN Face Detector for face detection. The model is only 2.7MB and is pretty fast even on the CPU. More details about the face detector can be found in our blog on Face Detection The face detection is done using the function getFaceBox as shown be

## Predict Gender

We will load the gender network into memory and pass the detected face through the network. The forward pass gives the probabilities or confidence of the two classes. We take the max of the two outputs and use it as the final gender prediction

## Predict Age

We load the age network and use the forward pass to get the output. Since the network architecture is similar to the Gender Network, we can take the max out of all the outputs to get the predicted age group

## Display Output

We will display the output of the network on the input images and show them using the imshow function

# Input Schema

```bash
python detect.py -img picture_name.jpeg
```

```bash
py detect.py -img picture_name.jpeg
```

# Output Result
example 1: 

<img width="415" alt="image" src="https://user-images.githubusercontent.com/72391917/161977331-4d769db2-7b6a-4ce0-ace5-bdbd0790b5f1.png">

