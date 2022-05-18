# Stalker

This repository is a ROS package implementing the project assigned for my Master thesis.
The goal of this project is to train a UAV using Reinforcement Learning and Computer Vision to follow 
a 'leader' which in our case is a wheeled robot (Summit XL).
The UAV is equipped with an onboard computer and a ZED stereocamera and is using the Ardupilot software.
The control scheme is applied to the attitude controller for better responsiveness and maneuverability.
The whole system is initially simulated in the Gazebo simulator using ROS and running SITL.
Training takes place in the simulation and then will be tested in real life conditions.

You should also need [this auxiliary repository](https://github.com/AndreasMit/vision.git) implementing the workspace for the Computer Vision tasks.

Folders explanation:
* in build.md you will find instructions on how to setup your system
* in run.md you will find instructions on how to start the simulation and run the scripts
* /models, /worlds : here you can find all the different environments and models in the simulated world, some of these models publish in topics in ROS. 
* /launch : here you find the launch files to start the simulations. Most times each of these is related to one .world file.
	* 	static_contour.launch : starts the world with the racetrack and the drone
	*   summit.launch  : starts the world with summit XL robot and the drone
	*	apm.launch : starts mavros and converts mavlink into ROS topics
	*   teleop.launch : starts joystick control to navigate the drone in the simulation
	* 	RLVS.launch : starts the RL agent and thus the training
* /scripts : you will find the different scripts for detection and training.
* /msg : ROS custom message types
* /analysis : different plots for better understanding
