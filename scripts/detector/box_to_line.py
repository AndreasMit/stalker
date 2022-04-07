#!/usr/bin/env python3 

import rospy
import numpy as np
from sensor_msgs.msg import Image
from color_detector.msg import PREDdata
from color_detector.msg import Line
from nav_msgs.msg import Odometry
from math import *

class line_detector:

	def __init__(self):
		self.box_sub = rospy.Subscriber("/box", PREDdata ,self.box_callback)
		self.att_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry,self.attitude_callback)
		self.line_pub = rospy.Publisher("/line_detector", Line, queue_size=10)
		# self.image_bub = rospy.Publisher("/RotDetection", Image, queue_size=10)
		self.phi = 0
		self.theta = 0
		self.x_velocity = 0

	def box_callback(self,box):
		mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
		mp_cartesian_v = self.featuresTransformation(mp_cartesian, self.phi, self.theta)
		mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)

		box = mp_pixel_v #edit
		lines = np.zeros(4)
    	lines[0] = np.linalg.norm(box[0] - box[1])
    	lines[1] = np.linalg.norm(box[1] - box[2])
    	lines[2] = np.linalg.norm(box[2] - box[3])
    	lines[3] = np.linalg.norm(box[3] - box[0])
    	long_line = np.argmax(lines) # we assume that the long line is always the one we want to follow 
    	angle = np.arctan2(box[long_line], box[(long_line+1)%4]) #returns [-pi,pi] , i want -pi/2 to pi/2 
    	angle = abs(angle) # [0,pi] 
    	angle -+ pi/2 # [-pi/2 , pi/2]
    	# cv.line(image, box[long_line], box[(long_line+1)%4], (0, 255, 0), 1)
    	angle += offset #adjust to your needs

    	box_center_x = (box[0][0]+box[2][0])//2 #center of diagonal
    	box_center_y = (box[0][1]+box[2][1])//2 
    	box_center = [box_center_x, box_center_y]
    	center = [480/2-1, 720/2-1]
    	distance = np.linalg.norm(np.array(center)-np.array(center_box))
    	# cv.line(image, center, box_center, (255, 0, 0), 1)
    	print(distance, angle, self.x_velocity)

	def attitude_callback(self, msg):
        self.x_velocity = msg.twist.twist.linear.x 
        print(self.x_velocity)
		quat = msg.pose.pose.orientation
		roll, pitch, yaw = self.quat2rpy(quat)
		self.phi = roll # roll -> phi
		self.theta = pitch # pitch -> theta

	def featuresTransformation(self, mp, phi, theta):       
    	Rphi = np.array([[1.0, 0.0, 0.0],[0.0, cos(phi), -sin(phi)],[0.0, sin(phi), cos(phi)]]).reshape(3,3)
    	Rtheta = np.array([[cos(theta), 0.0, sin(theta)],[0.0, 1.0, 0.0],[-sin(theta), 0.0, cos(theta)]]).reshape(3,3)
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

        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)  

        return roll, pitch, yaw   