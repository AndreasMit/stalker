## how to debug data taken from experiments

use rosbag to record and replay topics 

### record topic  (attitude/ odometry /action commands):
```
cd src/stalker/bagfiles
rosbag record -O subset /mavros/setpoint_raw/attitude /box /mavros/local_position/odom
```

### info about recorded topic
```
rosbag info <your bagfile>
```

### play recorded topic
```
rosbag play <your bagfile>
```

### visualize recorded topic
```
rosrun rqt_plot rqt_plot
```

### record all topics
```
rosbag record -a
```