# run:
cd ~/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris --console
roslaunch <package:iq_sim> <file.launch>
roslaunch darknet_ros darknet_ros.launch


rostopic list
rostopic echo /gazebo/model_states
rosmsg show nav_msgs/Odometry


MAVROS is a middle man which translates the MAVlink messages into ROS messages, which are easy to use and common between different robot systems. To start mavros run

roslaunch iq_sim apm.launch (set right address here so that you can get mavlink)
roslaunch mavros apm.launch

where to pubish attitude commands:
maybe interesting topic:
setpoint_attitude/attitude 
mavlink:
SET_ATTITUDE_TARGET 
