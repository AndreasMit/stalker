#!/usr/bin/env python3

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image, Imu
from color_detector.msg import PREDdata
from cv_bridge import CvBridge, CvBridgeError #convert Image to CV format
# from color_detector.msg import Line
from nav_msgs.msg import Odometry
import math
from math import *
import matplotlib.pyplot as plt



class line_detector:

	def __init__(self):
		self.box_sub = rospy.Subscriber("/box", PREDdata ,self.box_callback)
		self.att_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry,self.attitude_callback)
		self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
		self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw",Image,self.image_callback) 
		# self.line_pub = rospy.Publisher("/line_detector", Line, queue_size=10)
		self.image_pub = rospy.Publisher("/RotDetection", Image, queue_size=10)
		self.phi = 0
		self.theta = 0
		self.x_velocity = 0
		self.cu = 360.5
		self.cv = 240.5
		self.ax = 252.075
		self.ay = 252.075 
		self.Z = 5
		self.virtual_box = np.array([[0, 0], [0,0], [0,0], [0,0]])
		self.line = 0
		self.bridge = CvBridge()
		self.box_center = [0,0]
		self.real_center = [0,0]
		self.center = [int(720/2-1),int(480/2-1)]
		self.phi_imu, self.theta_imu, self.psi_imu = 0, 0, 0
		self.new = False
		self.dists = []
		self.rdists = []
		self.phis = []
		self.counter = 0

	def box_callback(self,box):
		if box.box_1==(0,0) and box.box_2==(0,0) and box.box_3==(0,0) and box.box_4==(0,0):
			print('out of bounds')
		else:
			if self.new == True:
				self.new = False
				mp = [ 	box.box_1[0],box.box_1[1], self.Z,
			 			box.box_2[0],box.box_2[1], self.Z,
			 			box.box_3[0],box.box_3[1], self.Z, 
			 			box.box_4[0],box.box_4[1], self.Z ]
				# print(mp)
				mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
				#use from imu and not from odometry
				mp_cartesian_v = self.featuresTransformation(mp_cartesian,  (-1)*self.theta_imu, (-1)*self.phi_imu) #Rx , Ry
				mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)

				self.virtual_box = np.array([ [mp_pixel_v[0],mp_pixel_v[1] ],
											[mp_pixel_v[3],mp_pixel_v[4] ],
											[mp_pixel_v[6],mp_pixel_v[7] ],
											[mp_pixel_v[9],mp_pixel_v[10]] ])
				self.virtual_box = np.int0(self.virtual_box)

				# print([self.virtual_box])
				lines = np.zeros(4)
				lines[0] = np.linalg.norm(self.virtual_box[0] - self.virtual_box[1])
				lines[1] = np.linalg.norm(self.virtual_box[1] - self.virtual_box[2])
				lines[2] = np.linalg.norm(self.virtual_box[2] - self.virtual_box[3])
				lines[3] = np.linalg.norm(self.virtual_box[3] - self.virtual_box[0])
				long_line = np.argmax(lines) 
				self.line = long_line
				# we assume that the long line is always the one we want to follow 
				# angle = np.arctan2(self.virtual_box[long_line], self.virtual_box[(long_line+1)%4]) #returns [-pi,pi] , i want -pi/2 to pi/2 
				x = self.virtual_box[long_line][0] - self.virtual_box[(long_line+1)%4][0]
				y = (self.virtual_box[long_line][1]-self.virtual_box[(long_line+1)%4][1])
				if(x!=0):
					angle = np.arctan(y/x) 
					angle = np.degrees(angle)
				else:
					angle = 90
				#90 is the best , 0 is the worst angle , good angle [-70,70], maybe change that later
				box_center_x = (self.virtual_box[0][0]+self.virtual_box[2][0])//2 #center of diagonal
				box_center_y = (self.virtual_box[0][1]+self.virtual_box[2][1])//2 
				self.box_center = [box_center_x, box_center_y]

				#test difference
				cx = (box.box_1[0]+box.box_3[0])//2
				cy = (box.box_1[1]+box.box_3[1])//2
				self.real_center = [cx, cy]
				# distance = np.linalg.norm(np.array(self.center)-np.array(self.box_center))
				distance_y = self.center[0]-self.box_center[0]
				distance_y_real = self.center[0]-self.real_center[0]

				distance_x = self.center[1]-self.box_center[1]
				distance_x_real = self.center[1]-self.real_center[1]

				# lower resolution because 1 pixel oscillates easily
				distance_y = distance_y//2
				# distance_y = round(distance_y)
				distance_y = distance_y*2
				# print(distance_y,distance_y_real ) #y axis on drone,  x axis on image plane	
				self.counter += 1
				self.dists.append(distance_y)
				self.phis.append(10*np.rad2deg(self.phi_imu))
				# self.rdists.append(distance_y_real)
				if (self.counter % 20 == 0):
					plt.plot(self.dists)
					plt.plot(self.phis)
					plt.grid()
					# plt.savefig('distances16.png')
				print(angle)
			# print(distance_y, angle)


	def image_callback(self,msg):
		try:
			image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
		except CvBridgeError as e:
			print(e)
		cv.drawContours(image, [self.virtual_box], 0, (0, 0, 255), 1)
		cv.line(image, tuple(self.virtual_box[self.line]), tuple(self.virtual_box[(self.line+1)%4]), (0, 255, 0), 1)
		# y-roll axis testing
		cv.line(image, tuple(self.center), (self.box_center[0], self.center[1]), (255, 0, 0), 2)
		cv.line(image, (self.center[0], self.center[1]-4), (self.real_center[0], self.center[1]-4), (0, 255, 0), 2)
		# x-pitch axis testing
		# cv.line(image, tuple(self.center), (self.center[0], self.box_center[1]), (255, 0, 0), 2)
		# cv.line(image, (self.center[0]-4, self.center[1]), (self.center[0]-4, self.real_center[1]), (0, 255, 0), 2)
		# show only dot of center
		# cv.line(image, (self.box_center[0], self.box_center[1]), (self.box_center[0]+2, self.box_center[1]), (255, 0, 0), 2)
		# cv.line(image, (self.real_center[0], self.real_center[1]), (self.real_center[0]+2, self.real_center[1]), (0, 255, 0), 2)
		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def imu_callback(self,msg):
		self.phi_imu, self.theta_imu, self.psi_imu = self.euler_from_quaternion(msg.orientation.x, msg.orientation.y,msg.orientation.z, msg.orientation.w)
		# distance_y = self.center[0]-self.box_center[0]
		#show only when roll and pitch are updated otherwise it oscillates
		self.new = True

	def attitude_callback(self, msg):
		return
		# self.x_velocity = msg.twist.twist.linear.x 
		# self.z_position = msg.pose.pose.position.z
		# quat = msg.pose.pose.orientation
		# #roll pitch yaw are returned correct -> phi and theta maybe was the problem
		# roll, pitch, yaw = self.quat2rpy(quat) #return angles in rads
		# # self.phi = roll # roll -> phi
		# # self.theta = pitch # pitch -> theta
		# self.phi = pitch 
		# self.theta = roll 
		# # print(self.x_velocity, np.rad2deg(roll), np.rad2deg(pitch) , self.z_position)

		#phi and theta need to be in rads
	def featuresTransformation(self, mp, phi, theta):       
		Rphi = np.array([[1.0, 0.0, 0.0],
						[0.0, cos(phi), -sin(phi)],
						[0.0, sin(phi), cos(phi)]]).reshape(3,3)
		Rtheta = np.array([[cos(theta), 0.0, sin(theta)],
						[0.0, 1.0, 0.0],
						[-sin(theta), 0.0, cos(theta)]]).reshape(3,3)
		Rft = np.dot(Rphi, Rtheta)
		mpv0 = np.dot(Rft, mp[0:3])
		mpv1 = np.dot(Rft, mp[3:6])
		mpv2 = np.dot(Rft, mp[6:9])
		mpv3 = np.dot(Rft, mp[9:12])
		mpv = np.hstack((mpv0, mpv1, mpv2, mpv3))    
		return mpv

	def cartesian_from_pixel(self, mp_pixel, cu, cv, ax, ay):
		Z_0 = mp_pixel[2]
		X_0 = Z_0*((mp_pixel[0]-cu)/ax)
		Y_0 = Z_0*((mp_pixel[1]-cv)/ay)
		  
		Z_1 = mp_pixel[5]
		X_1 = Z_1*((mp_pixel[3]-cu)/ax)
		Y_1 = Z_1*((mp_pixel[4]-cv)/ay)
   
		Z_2 = mp_pixel[8]
		X_2 = Z_2*((mp_pixel[6]-cu)/ax)
		Y_2 = Z_2*((mp_pixel[7]-cv)/ay)

		Z_3 = mp_pixel[11]    
		X_3 = Z_3*((mp_pixel[9]-cu)/ax)
		Y_3 = Z_3*((mp_pixel[10]-cv)/ay)

		mp_cartesian = np.array([X_0, Y_0, Z_0, X_1, Y_1, Z_1, X_2, Y_2, Z_2, X_3, Y_3, Z_3])  
		return mp_cartesian

	def pixels_from_cartesian(self, mp_cartesian, cu, cv, ax, ay):
		u_0 = (mp_cartesian[0]/mp_cartesian[2])*ax + cu
		v_0 = (mp_cartesian[1]/mp_cartesian[2])*ay + cv
	   
		u_1 = (mp_cartesian[3]/mp_cartesian[5])*ax + cu
		v_1 = (mp_cartesian[4]/mp_cartesian[5])*ay + cv

		u_2 = (mp_cartesian[6]/mp_cartesian[8])*ax + cu
		v_2 = (mp_cartesian[7]/mp_cartesian[8])*ay + cv
	
		u_3 = (mp_cartesian[9]/mp_cartesian[11])*ax + cu
		v_3 = (mp_cartesian[10]/mp_cartesian[11])*ay + cv

		mp_pixel = np.array([u_0, v_0, mp_cartesian[2], u_1, v_1, mp_cartesian[5], u_2, v_2, mp_cartesian[8], u_3, v_3, mp_cartesian[11]])        
		return mp_pixel

	# Convert quaternion to roll,pitch,yaw (degrees)
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

		# roll = np.rad2deg(roll)
		# pitch = np.rad2deg(pitch)
		# yaw = np.rad2deg(yaw)  

		return roll, pitch, yaw   

 
	def euler_from_quaternion(self,x, y, z, w):
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
	 
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)
	 
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
		# in radians
		return roll_x, pitch_y, yaw_z 


if __name__=='__main__':
	rospy.init_node('box_to_line', anonymous=True)
	line_detector()
	while not rospy.is_shutdown:
		r.sleep()    

	rospy.spin() 