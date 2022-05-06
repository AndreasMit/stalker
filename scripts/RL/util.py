#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Quaternion
import math


# Convert roll, pitch, yaw (in degrees) to quaternion
def rpy2quat(roll,pitch,yaw):
    
    q = Quaternion()
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)

    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q   

# Convert quaternion to roll,pitch,yaw (degrees)
def quat2rpy(quat):

    sinr_cosp = 2.0*(quat.w*quat.x + quat.y*quat.z)
    cosr_cosp = 1 - 2*(quat.x*quat.x + quat.y*quat.y)
    roll = math.atan2(sinr_cosp , cosr_cosp)    

    sinp = 2*(quat.w*quat.y - quat.z*quat.x)
    if abs(sinp)>=1:
        pitch = math.pi/2.0 * sinp/abs(sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2*(quat.w*quat.z + quat.x*quat.y)
    cosy_cosp = 1 - 2*(quat.y*quat.y + quat.z*quat.z)
    yaw = math.atan2(siny_cosp,cosy_cosp)

    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)  

    return roll, pitch, yaw 