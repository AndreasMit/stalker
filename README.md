# Stalker

This is the repository for my Masters Diploma project.
In this project i train a drone to follow a wheeled robot using Reingorcement Learning.
Drone is running the Ardupilot software while an onboard computer with a ZED camera provide 
attitude commands via mavlink.
The whole system is initially simulated in Gazebo using ROS and running SITL.
Training takes place in simulation and then it is going to be tested in real life conditions.


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