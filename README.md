# Virtual keyboard based on hand gestures
Here we propose a hand gesture recognition model that uses a single frame without using any temporal information. Based on the previous observations we used deep learning for hand gesture recognition. The proposed model as a whole can be considered as two main parts, the feature extractor, and the classifier. 

## Feature extraction
For the feature extraction part we used pretrained model of CPM implemented in tensorflow. 
This is the **Tensorflow** implementation of [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release), one of the state-of-the-art models for **2D body and hand pose estimation**.

<p align = "center" >
     <img src="https://github.com/kc07ce/virtual_keyboard-based-on-hand-gestures/blob/master/cpm_hand.gif", width = "480">
</p>

## Classification

We used Massey Universitie's ASL dataset and extracted the coordinates of 21 joints from CPM for each image. These coordinates are then fed to a MLP consisting of 9 layers and trained it.

<p align = "center" >
     <img src="https://github.com/kc07ce/virtual_keyboard-based-on-hand-gestures/blob/master/massey_static.png", width = "480">
</p>

### Run demo scripts
To test our code with a live demo, download the weights of our [classifier](https://drive.google.com/file/d/1cWPojr2SVMl8WjWsgOuFd9ETwBR-dQod/view?usp=sharing) and [CPM model](https://drive.google.com/file/d/1p4FJB0hVR4YrSku-3c9u2ej8A17ORcKy/view?usp=sharing). And place these weights in the " models/weights/ "  folder and run the "demo_virtual_keyboard.py".

As we didn't add any hand detector, try to place the hand in the middle of the frame and make sure that you get a proper skeleton of the shown gesture for better classification.
