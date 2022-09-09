This is code from NYU hand pose dataset

Overview

The NYU Hand pose dataset contains 8252 test-set and 72757 training-set frames of captured RGBD data with ground-truth hand-pose information. For each frame, the RGBD data from 3 Kinects is provided: a frontal view and 2 side views. The training set contains samples from a single user only (Jonathan Tompson), while the test set contains samples from two users (Murphy Stein and Jonathan Tompson). A synthetic re-creation (rendering) of the hand pose is also provided for each view.

We also provide the predicted joint locations from our ConvNet (for the test-set) so you can compare performance. Note: for real-time prediction we used only the depth image from Kinect 1.

The source code to fit the hand-model to the depth frames here can be found here

NEW: The dataset used to train the RDF is also public! It contains 6736 depth frames of myself doing various hand gesture (seated and standing) and the ground truth per-pixel labels (hand/not hand).

https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#overview

@article{tompson14tog,
  author = {Jonathan Tompson and Murphy Stein and Yann Lecun and Ken Perlin}
  title = {Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks,
  journal = {ACM Transactions on Graphics},
  year = {2014},
  month = {August},
  volume = {33}
}
