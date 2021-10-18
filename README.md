# Webcam Mask Detection with Transfer Learning

## Introduction
The goal of this project is to train a Convolutional Neural Network that detects if a person is using or not a facial mask. After that, we will create a python application that runs our model in real-time on the video provided by our webcam. 

By doing so, we will explore the concepts of Transfer Learning and Data Augmentation.

In this README file, we briefly introduce the topics, make sure to read the commented jupyter notebook and python file if you want to understand the code. 

## What is Transfer Learning
Transfer learning focuses on applying the knowledge gained while solving one problem into a different but related problem. From a practical view, that means that we will use an already-trained neural network to solve a new problem, instead of training it from scratch we will train just the last layers. 

<img src="/Users/cauaveiga/Documents/Coding/Coursera AI Engineering/Mask Classifier/Videos/Transfer Learning.png"
     alt="Markdown Monster icon"
     width="1024" 
     height="512"
     style="vertical-align:middle;margin:0px 0px" />

The benefit of using transfer learning is obvious time complexity and dataset size. Once we have access to an already trained deep neural network, that was trained in a gigantic dataset, we can use the knowledge already learned and retrain the last layer with a new and more limited dataset, which will be much faster than training from scratch and will, of course, require less computation power.

## What is Data Augmentation

Sometimes we don´t have a large enough dataset to properly train our neural network, in those cases, data augmentation comes in handy as a solution to this problem (gathering more data is not always possible or plausible).

Data augmentation consists of techniques used to increase the dataset by slightly modifying the existing data (you can also create a synthetic dataset, but it is not in the scope of this project). The techniques consist of images transformations, such as geometric transformations, flipping, color modification, cropping, etc. 

(INSERT EXAMPLE IMAGE HERE)

## The datset and pre-processing pipeline

The dataset consists of labeled data of persons with a mask and without a mask, it was provided in https://github.com/prajnasb/observations/tree/master/experiements.
We split the data into a training set with 70% of the images and a test set with the other 30%.

Our preprocessing pipeline starts by passing the data into the TensorFlow [preprocess_input()](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input). After that, we transform our labels using OneHotEncoder method, which will transform our categorical data (maks,without_mask) into binary categorical data (1,0). Finally, we define the image generator for the training data augmentation, where we use rotation, width shift, height shift, horizontal flip, and zoom.  

## Training the model

We start by loading the MobileNetV2 pre-trained network and constructing the head of the model that we will replace. We then loop over all layers in the base model and freeze them so they are not updated during the first training process. 

After we proceed to the training routine, with a total of 40 epochs, using Adam optimizer and binary cross-entropy. 

Once the training is complete, we make predictions on the test set and evaluate our model performance:

<img src="/Users/cauaveiga/Documents/Coding/Coursera AI Engineering/Mask Classifier/Videos/Model Val.png"
     alt="Markdown Monster icon"
     width="" 
     height="512"
     style="vertical-align:middle;margin:0px 0px" />

To finish, we save the model so we can load it in the webcam app.

## Applying the model into real time webcam application using OpenCv

We use the openCv ['haarcascade_frontalface'](https://github.com/opencv/opencv/tree/master/data/haarcascades)  Cascade Classifier to make facial detections, after that we pass the cropped image of the detected face through our model and predict if the person is using a facial mask or not.

# Results
A demonstration of the script working in real time!

<img src="/Users/cauaveiga/Documents/Coding/Coursera AI Engineering/Mask Classifier/Videos/IMG_8316.GIF"
     alt="Markdown Monster icon"
     width="" 
     height="512"
     style="vertical-align:middle;margin:0px 0px" />


Notice that the confidence of the model is lower when the mask is on the chin, and when it is totally off or totally on is easier for the model to make a prediction.