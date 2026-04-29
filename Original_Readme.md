## Hand Pose Estimation ##

The purpose of this project is to estimate the pose of a hand in an image. The project uses a convolutional neural network (CNN) or deep learning model like yolo or pose estimation to identify the user's hand and generate the coordinate system. The model is trained to predict the 3D coordinates of the hand joints from a single RGB image.

in terms of the dataset and the images, we are using the images of "right hand rule" to train the model. The right hand rule here refers to the convention used to define the coordinate system. The thumb is the x-axis, the index finger is the y-axis, and the middle finger is the z-axis. The dataset consists of image of hand in various pose and orientations. The images are labeled with the corresponding hand joint coordinates, which are used as ground truth for training the model. The dataset is split into training and testing sets to evaluate the performance of the model.

What do the models do? 

The model will need to do the following 
* identify the hand in the image/videos 
* segment the hand from the background
* estimate the 3D coordinates of the hand joints



Moreover, in terms of the hand pose estimation, the project can be used for various applications such as virtual reality, augmented reality, and human-computer interaction. The estimated hand pose can be used to control virtual objects, interact with virtual environments, or even for sign language recognition.