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
cd ~/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
```
```
cd ~/ardupilot/ArduCopter/ && sim_vehicle.py --mavproxy-args="--streamrate=30" -v ArduCopter -f gazebo-iris
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
## run detection scripts:
```
conda activate aerials-env
cd ~/vision
source devel/setup.bash
rosrun color_detector detect_line.py
```
## test box to line conversion:
```
conda activate aerials-env
cd ~/vision
source devel/setup.bash
rosrun color_detector box_to_line.py
```
## run RL:
```
conda activate aerials-env
rosrun stalker RLVS.py
```
## show detection
```
rosrun image_view image_view image:=/Detection
```
```
rosrun image_view image_view image:=/RotDetection
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
rosrun image_view video_recorder image:=/iris_demo/ZED_stereocamera/camera/left/image_raw _filename:='video.avi'
```
```
rosrun image_view image_view image:=/iris_demo/ZED_stereocamera/camera/left/image_raw
```
```
rosrun image_view video_recorder image:=/iris_demo/ZED_stereocamera/camera/left/image_raw _filename:='b7588637-aae8-4803-a9c3-edf4bcea43b6/try.avi'
```
to use your controller: 
```
roslaunch stalker teleop.launch teleop_args:=-vel
```



