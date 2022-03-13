

# install ardupilot:

git git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git checkout Copter-4.2
git submodule update --init --recursive

# build vehicle simulator (sitl):

cd ~/ardupilot/ArduCopter
sim_vehicle.py -w

# install gazebo:

sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install gazebo9 libgazebo9-dev

# install ros:

http://wiki.ros.org/melodic/Installation/Ubuntu

# create catkin workspace:

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
cd ~/catkin_ws
wstool init ~/catkin_ws/src

# install ros dependencies

rosinstall_generator --upstream mavros | tee /tmp/mavros.rosinstall
rosinstall_generator mavlink | tee -a /tmp/mavros.rosinstall
wstool merge -t src /tmp/mavros.rosinstall
wstool update -t src
rosdep install --from-paths src --ignore-src --rosdistro `echo $ROS_DISTRO` -y
catkin build
sudo ~/catkin_ws/src/mavros/mavros/scripts/install_geographiclib_datasets.sh


# install stalker package:
cd ~/catkin_ws/src
git clone https://github.com/AndreasMit/stalker
cd ~/catkin_ws
catkin build


# install yolo:
cd ~/catkin_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
catkin build -DCMAKE_BUILD_TYPE=Release 
//-DCMAKE_C_COMPILER=/usr/bin/gcc-6

edit darknet_ros/darknet_ros/config
/camera/rgb/image_raw -> /camera/image_raw
edit darknet_ros.launch
<arg name="network_param_file"         default="$(find darknet_ros)/config/yolov2-tiny.yaml"/>


# install gazebo models:
git clone https://github.com/osrf/gazebo_models.git

# edit ~/.bashrc
echo "export PATH=$PATH:$HOME/ardupilot/Tools/autotest" >> ~/.bashrc
echo "PATH=/usr/lib/ccache:$PATH" >> ~/.bashrc
echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=/home/rico/catkin_ws/src/stalker/models:/home/rico/gazebo_models" >> ~/.bashrc
source ~/.bashrc







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

echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=~/ardupilot_gazebo/models' >> ~/.bashrc
. ~/.bashrc


run:
gazebo --verbose ~/ardupilot_gazebo/worlds/iris_arducopter_runway.world
cd ~/ardupilot/ArduCopter/
sim_vehicle.py -v ArduCopter -f gazebo-iris --console
###############################	









