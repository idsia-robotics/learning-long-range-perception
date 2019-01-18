# Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and Odometry

*Mirko Nava, Jérôme Guzzi, R. Omar Chavez-Garcia, Luca M. Gambardella and Alessandro Giusti*

*Robotics Lab, Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)*

We introduce a general self-supervised approach to predict the future outputs of a short-range sensor (such as a proximity sensor) given the current outputs of a long-range sensor (such as a camera); we assume that the former is directly related to some piece of information to be perceived (such as the presence of an obstacle in a given position), whereas the latter is information-rich but hard to interpret directly.
We instantiate and implement the approach on a small mobile robot to detect obstacles at various distances using the video stream of the robot's forward-pointing camera, by training a convolutional neural network on automatically-acquired datasets.  We quantitatively evaluate the quality of the predictions on unseen scenarios, qualitatively evaluate robustness to different operating conditions, and demonstrate usage as the sole input of an obstacle-avoidance controller.
We additionally instantiate the approach on a different simulated scenario with complementary characteristics, to exemplify the
generality of our contribution.

![Predictions](https://github.com/idsia-robotics/Learning-Long-range-Perception/blob/master/img/predictions.png "Predictions")
*Prediction of a model trained with the proposed approach applied on a camera mounted on a Mighty Thymio **(a)**, on a TurtleBot **(b)** and on the belt of a person **(c)**.*

![Predictions](https://github.com/idsia-robotics/Learning-Long-range-Perception/blob/master/img/simulation_results.png "Predictions")
*Simulation setup and results of the proposed approach applied on 3 cameras mounted on a Pioneer 3AT with different rotations.
**Left & Center-left**: robot setup with cameras' views.
**Center-right**: number of extracted known labels from 70min of recording.
**Right**: achieved AUC score of a model trained from 35min of recording.*

The preprint PDF of the article is available at the following link [arXiv:1809.07207](https://arxiv.org/abs/1809.07207).

### Bibtex

```properties  
@article{nava2019learning,
  title={Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and Odometry},
  author={Nava, Mirko and Guzzi, Jerome and Chavez-Garcia, Omar and Gambardella, Luca and Giusti, Alessandro},
  journal="IEEE Robotics and Automation Letters",
  year="2019"
}
```

### Videos

All the video material of models trained with the proposed approach on different scenarios, robots and systems is available [here](https://github.com/idsia-robotics/Learning-Long-range-Perception/tree/master/video).

### Datasets

The real world dataset is available at [this link](https://drive.switch.ch/index.php/s/v6P93gv6lA77AQ4).
It is stored as an HDF5 file containing two groups per recording called respectively *"bag{index}_x"* and *"bag{index}_y"* for a total of 11 recordings (22 groups).

The simulation dataset is available at [this link](https://drive.switch.ch/index.php/s/oYZCNfIeZ06mKT0).
It is stored as an HDF5 file containing a main group per recording called *"bag{index}"*. Each main group is divided into subgroups *"/x"* and *"/y"* that are respectively divded into *"/input_cam1", "/input_cam2", "/input_cam3"* and *"/output_target1"*.

### Code

The entire codebase is avaliable [here](https://github.com/idsia-robotics/Learning-Long-range-Perception/tree/master/code).
In order to generate the datasets, of which download links are present above, one should launch the script preprocess.py which will create the dataset in hdf5 file format, starting from a collection of ROS bagfiles stored in a given folder.

The script train.py is used to train the model, which is defined in unified_model.py, using a given hdf5 dataset. A list of the available parameters can be seen by launching  `python train.py -h `.

The script test.py is used to test the model, which is defined in unified_model.py, using a subset of the hdf5 groups defined in the script. A list of the available parameters can be seen by launching  `python test.py -h `.

The scripts visualize.py and visualize_output.py are respectively used to visualize the real world dataset collected consisting in the camera's view and the ground truth labels, and to visualize the same information plus the selected models' prediction.
