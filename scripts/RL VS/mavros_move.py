#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
import math
import time
from mavros_msgs.msg import PositionTarget


class Move:
	def __init__(self):
		self.pub_pos = rospy.Publisher("/mavros/setpoint_raw/local",PositionTarget,queue_size=10000)
		self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)
		self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.move)
		self.count = 0
	def move(self, direction):
		if self.count <=10:
			print('lets yaw')
			roll_des = 0
			pitch_des = 0
			yaw_des = 1
			action_mavros = AttitudeTarget()
			action_mavros.type_mask = 7
			action_mavros.thrust = 0.5
			action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,yaw_des)
			self.pub_action.publish(action_mavros)
			self.count = self.count+1

	def rpy2quat(self,roll,pitch,yaw):
		
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


if __name__=='__main__':
	rospy.init_node('mavros_move', anonymous=True)
	Move()
	while not rospy.is_shutdown:
		r.sleep()    

	rospy.spin()