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

	def move(self, msg):
		if self.count == 0:
			action_mavros = AttitudeTarget()
			action_mavros.type_mask = 7
			action_mavros.thrust = 0.5
			action_mavros.orientation = self.rpy2quat(0.0,0.0,90.0)
			self.pub_action.publish(action_mavros)
			self.count = self.count + 1
		quat = msg.pose.pose.orientation
		roll, pitch, yaw = self.quat2rpy(quat)
		self.x_velocity = msg.twist.twist.linear.x 
		self.z_position = msg.pose.pose.position.z
		print(roll, pitch , yaw)
		# roll_des = 0.0
		# pitch_des = 0.0
		# yaw_des = 90.0
		# #roll and pitch work just fine
		# if self.count ==20:
		# 	print('right')
		# 	roll_des = 5.0 
		# if self.count ==40:
		# 	print('left')
		# 	roll_des = -7.0

		# action_mavros = AttitudeTarget()
		# action_mavros.type_mask = 7
		# action_mavros.thrust = 0.5
		# action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,yaw_des)
		# self.pub_action.publish(action_mavros)
		# self.count = self.count+1
		

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

	def quat2rpy(self,quat):

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


if __name__=='__main__':
	rospy.init_node('mavros_move', anonymous=True)
	Move()
	while not rospy.is_shutdown:
		r.sleep()    

	rospy.spin()