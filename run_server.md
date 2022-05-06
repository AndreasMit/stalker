# run on server 	

## connent to remote
```
ssh andreas@147.102.51.87
``` 
with graphics
```
ssh -X andreas@147.102.51.87
```

## start world:
```
cd  ~/andreas
roslaunch stalker static_contour.launch
```
or
```
cd  ~/andreas
roslaunch stalker summit.launch
```
## start ardupilot sitl:
```
cd ~/andreas/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris
```
## start mavros (convert mavlink to ROS topics):
```
cd ~/andreas
roslaunch stalker apm.launch
```
## start your script:
```
cd ~/andreas
roslaunch stalker RLVS.launch
```
or
```
cd ~/andreas
rosrun stalker RLVS.py
```
## run detection scripts:
```
conda activate tf-gpu-cuda10
cd ~/andreas/vision
source devel/setup.bash
rosrun color_detector detect_line.py
```
## run RL:
```
conda activate tf-gpu-cuda10
cd ~/andreas
rosrun stalker RLVS.py
```
## show detection
```
rosrun image_view image_view image:=/Detection
```
```
rosrun image_view image_view image:=/RotDetection
```

# .bashrc changes
the following lines were added in .bashrc:
```
export PATH=$PATH:/home/andreas/andreas/ardupilot/Tools/autotest
export PATH=/usr/lib/ccache:$PATH

source /usr/share/gazebo/setup.sh
source /opt/ros/melodic/setup.bash
source ~/andreas/catkin_ws/devel/setup.bash

export GAZEBO_MODEL_PATH=~/andreas/catkin_ws/src/stalker/models

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/sotiris/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/sotiris/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/sotiris/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/sotiris/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```