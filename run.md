# How to run the simulation:
## Start your world:
### static contour:
```
roslaunch stalker static_contour.launch
```
### following:
```
roslaunch stalker summit.launch
```
## start ardupilot sitl:
```
cd ~/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris --console
```
## start mavros (convert mavlink to ROS topics):
```
roslaunch stalker apm.launch
```
## start your script:
```
roslaunch stalker RLVS.launch
```
or
```
rosrun stalker RLVS.py
```
## extra (for reference):
start gazebo alone:
```
gazebo --verbose ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world
```
launch world with summit:
```
roslaunch summit_xl_sim_bringup summit_xl_complete.launch
```
general case:
```
roslaunch <package> <file.launch>
```
```
rosrun image_view video_recorder image:=/iris_demo/ZED_stereocamera/camera/left/image_raw [image transport type]
```
```
rosrun image_view video_recorder image:=/iris_demo/ZED_stereocamera/camera/left/image_raw _filename:='b7588637-aae8-4803-a9c3-edf4bcea43b6/try.avi'
```
```
roslaunch mavros_extras teleop.launch teleop_args:=-vel
```
///
rostopic list
rostopic echo /gazebo/model_states
rosmsg show nav_msgs/Odometry

#publish to :
/mavros/setpoint_raw/local
/mavros/setpoint_raw/attitude

#subscribe to :
/iris_demo/ZED_stereocamera/camera/left/image_raw


where to pubish attitude commands:
setpoint_attitude/attitude 
mavlink:
SET_ATTITUDE_TARGET 



