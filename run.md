# run:
cd ~/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris --console
roslaunch <package:iq_sim> <file.launch>
roslaunch darknet_ros darknet_ros.launch

gazebo --verbose ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world

roslaunch summit_xl_sim_bringup summit_xl_complete.launch

# static contour:

roslaunch stalker static_contour.launch

# following

roslaunch stalker summit.launch

///
rostopic list
rostopic echo /gazebo/model_states
rosmsg show nav_msgs/Odometry

roslaunch iq_sim apm.launch (set right address here so that you can get mavlink)
roslaunch mavros apm.launch

where to pubish attitude commands:
maybe interesting topic:
setpoint_attitude/attitude 
mavlink:
SET_ATTITUDE_TARGET 
