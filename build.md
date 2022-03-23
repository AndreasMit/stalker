# instructions how to setup your system so that you can use the simulator

## dependencies #1:
```
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install python-matplotlib python-serial python-wxgtk3.0 python-wxtools python-lxml python-scipy python-opencv ccache gawk python-pip python-pexpect
sudo pip install future pymavlink MAVProxy
sudo apt install curl 
```
## install ardupilot:
```
cd ~
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git checkout Copter-4.2
git submodule update --init --recursive
```
## update ~/.bashrc #1:
```
echo "export PATH=$PATH:$HOME/ardupilot/Tools/autotest" >> ~/.bashrc
echo "export PATH=/usr/lib/ccache:$PATH" >> ~/.bashrc
source ~/.bashrc
```
## build vehicle simulator (sitl):
```
cd ~/ardupilot/ArduCopter
python -m pip install empy
sim_vehicle.py -w
```
## install gazebo:
```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install gazebo9 libgazebo9-dev
```
## update ~/.bashrc #2:
```
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
source ~/.bashrc
```
## install ros:
follow instructions on: http://wiki.ros.org/melodic/Installation/Ubuntu
or
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-melodic-desktop-full
```
## update ~/.bashrc #3:
```
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
## install dependencies #2:
```
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```
## initialize rosdep:
```
sudo rosdep init
rosdep update

```
## install dependencies #3:
```
sudo apt-get install python-wstool python-rosinstall-generator python-catkin-tools
sudo apt install ros-melodic-gazebo-ros ros-melodic-gazebo-plugins
```
## create catkin workspace:
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
```
## install mavros and mavlink:
```
cd ~/catkin_ws
wstool init ~/catkin_ws/src
rosinstall_generator --upstream mavros | tee /tmp/mavros.rosinstall
rosinstall_generator mavlink | tee -a /tmp/mavros.rosinstall
wstool merge -t src /tmp/mavros.rosinstall
wstool update -t src
rosdep install --from-paths src --ignore-src --rosdistro `echo $ROS_DISTRO` -y
catkin build
sudo ~/catkin_ws/src/mavros/mavros/scripts/install_geographiclib_datasets.sh
```
## update ~/.bashrc #4:
```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
## install stalker package:
```
cd ~/catkin_ws/src
git clone https://github.com/AndreasMit/stalker
cd ~/catkin_ws
catkin build
```
## get gazebo models(maybe needed):
```
cd ~
git clone https://github.com/osrf/gazebo_models.git
```
## update ~/.bashrc #5:
```
echo "export GAZEBO_MODEL_PATH=/home/rico/catkin_ws/src/stalker/models:/home/rico/gazebo_models" >> ~/.bashrc
source ~/.bashrc
```
## install yolo (optional):
```
cd ~/catkin_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
catkin build -DCMAKE_BUILD_TYPE=Release 
//-DCMAKE_C_COMPILER=/usr/bin/gcc-6
```

edit darknet_ros/darknet_ros/config
/camera/rgb/image_raw -> /camera/image_raw
edit darknet_ros.launch
<arg name="network_param_file"         default="$(find darknet_ros)/config/yolov2-tiny.yaml"/>


######  i dont think i need this ############
cd ~
git clone https://github.com/khancyr/ardupilot_gazebo.git
cd ardupilot_gazebo
git checkout dev

mkdir build
cd build
cmake ..
make -j4
sudo make install

echo 'export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models' >> ~/.bashrc
. ~/.bashrc

###############################	









