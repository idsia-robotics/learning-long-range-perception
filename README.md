# Learning Long-range Perception using Self-Supervision from Short-Range Sensors and Odometry

*Mirko Nava, Jérôme Guzzi, R. Omar Chavez-Garcia, Luca M. Gambardella and Alessandro Giusti*

We introduce a general self-supervised approach to predict the future outputs of a short-range sensor (such as a proximity sensor) given the current outputs of a long-range sensor (such as a camera);
we assume that the former is directly related to some piece of information to be perceived (such as the presence of an obstacle in a given position), whereas the latter is information-rich but hard to interpret directly.
We instantiate and implement the approach on a small mobile robot to detect obstacles at various distances using the video stream of the robot's forward-pointing camera, by training a convolutional neural network on automatically-acquired datasets.
We quantitatively evaluate the quality of the predictions on unseen scenarios, qualitatively evaluate robustness to different operating conditions, and demonstrate usage as the sole input of an obstacle-avoidance controller.

![Robots](https://github.com/Mirko-Nava/Learning-Long-range-Perception/blob/master/img/robots.jpg "Robots")

### Videos

Sample videos of the model's prediction, the camera's view and the target labels are available [here](https://github.com/Mirko-Nava/Learning-Long-range-Perception/tree/master/video).

### Dataset

The whole collected dataset is available at [this link](https://drive.switch.ch/index.php/s/v6P93gv6lA77AQ4).
It is stored as an HDF5 file containing two groups per recording called respectively *bag{index}_x* and *bag{index}_y* for a total of 11 recordings (22 groups).
