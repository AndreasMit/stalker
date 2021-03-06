# instructions how to setup the system

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
copy these lines in ~/.bashrc:
```
export PATH=$PATH:$HOME/ardupilot/Tools/autotest
export PATH=/usr/lib/ccache:$PATH
```
```
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
## get gazebo models(optional):
```
cd ~
git clone https://github.com/osrf/gazebo_models.git
```
## update ~/.bashrc #5:
```
echo "export GAZEBO_MODEL_PATH=~/catkin_ws/src/stalker/models:~/gazebo_models" >> ~/.bashrc
source ~/.bashrc
```
## install Summit XL package to use in sim:
```
sudo apt-get install -y python3-vcstool
cd ~/catkin_ws
vcs import --input https://raw.githubusercontent.com/RobotnikAutomation/summit_xl_sim/melodic-devel/repos/summit_xl_sim_devel.repos
rosdep install --from-paths src --ignore-src --skip-keys="summit_xl_robot_control marker_mapping robotnik_locator robotnik_pose_filter robotnik_gazebo_elevator" -y
catkin build
```

## Scripts and python environments
## install conda and create virtual environments:
follow instructions: https://docs.anaconda.com/anaconda/install/linux/
create an enviroment named 'aerials':
```
conda create -n aerials tensorflow-gpu=1.14 cudatoolkit=10.0 python=3.6
```
to start virtual environment:
```
conda activate aerials-env
```
install python dependencies:
```
conda install -c conda-forge keras=2.2.5
pip install keras-segmentation
pip install numpy
pip install scipy matplotlib pillow
pip install imutils h5py==2.10.0 requests progressbar2
pip install cython
pip install scikit-learn scikit-build scikit-image
pip install opencv-contrib-python==4.4.0.46
pip install tensorflow-gpu==1.14.0
pip install keras==2.2.5
pip install opencv-python==4.4.0.42
pip install keras-segmentation
pip install rospkg empy
pip install matplotlib
pip install 'gast==0.2.2'
pip install opencv-python-headless==4.1.2.30
```
to stop virtual environment:
```
conda deactivate
```
also deactivate base environment:
```
conda config --set auto_activate_base False
```
## make scritps executable:
```
cd ~/catkin_ws/src/stalker/scripts
chmod +x RLVS.py
```
This command installs all the packages that the packages in your catkin workspace depend upon but are missing on your computer: (just for reference)
```
rosdep install --from-paths src --ignore-src -r -y :
```

## YOLO installation
```
sudo apt install nvidia-cuda-toolkit

cd catkin_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
```
```
cd catkin_ws
catkin build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-6
```