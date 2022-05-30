#!/usr/bin/env python3 

import rospy
import numpy as np
import math
from math import *

class center_detector:

	def __init__(self):		
		self.cu = 360.5
		self.cv = 240.5
		self.ax = 252.075
		self.ay = 252.075 
		self.center = [int(720/2-1),int(480/2-1)]

	#the box and roll,pitch has to be synchronized in order to have accurate transforms
	def compute(self, box, roll, pitch, Z):

		if box.box_1==(0,0) and box.box_2==(0,0) and box.box_3==(0,0) and box.box_4==(0,0):
			distance_y = 10000
			distance_x = 10000
			angle = 0
			return distance_x, distance_y, angle 

		else:

			mp = [ 	box.box_1[0],box.box_1[1], Z,
		 			box.box_2[0],box.box_2[1], Z,
		 			box.box_3[0],box.box_3[1], Z, 
		 			box.box_4[0],box.box_4[1], Z ]
			# print(mp)
			Rotx = (-1)*np.deg2rad(pitch) #theta_imu, rotx on camera
			Roty = (-1)*np.deg2rad(roll) #phi imu, roty on camera

			mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
			mp_cartesian_v = self.featuresTransformation(mp_cartesian, Rotx, Roty) # x,y on camera coordinates
			mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)

			virtual_box = np.array([ [mp_pixel_v[0],mp_pixel_v[1] ],
										[mp_pixel_v[3],mp_pixel_v[4] ],
										[mp_pixel_v[6],mp_pixel_v[7] ],
										[mp_pixel_v[9],mp_pixel_v[10]] ])
			virtual_box = np.int0(virtual_box)

			# print([self.virtual_box])
			lines = np.zeros(4)
			lines[0] = np.linalg.norm(virtual_box[0] - virtual_box[1])
			lines[1] = np.linalg.norm(virtual_box[1] - virtual_box[2])
			lines[2] = np.linalg.norm(virtual_box[2] - virtual_box[3])
			lines[3] = np.linalg.norm(virtual_box[3] - virtual_box[0])
			long_line = np.argmax(lines) 
			# we assume that the long line is always the one we want to follow 
			# angle = np.arctan2(self.virtual_box[long_line], self.virtual_box[(long_line+1)%4]) #returns [-pi,pi] , i want -pi/2 to pi/2 
			x = virtual_box[long_line][0] - virtual_box[(long_line+1)%4][0]
			y = virtual_box[long_line][1] - virtual_box[(long_line+1)%4][1]
			if(x!=0):
				angle = np.arctan(y/x) 
				angle = np.degrees(angle)
			else:
				angle = 90
			#convert to 
			angle = np.sign(angle)*(90 - abs(angle))

			box_center_x = (virtual_box[0][0]+virtual_box[2][0])//2 #center of diagonal
			box_center_y = (virtual_box[0][1]+virtual_box[2][1])//2 
			box_center = [box_center_x, box_center_y]
			
			distance_x = self.center[1]-box_center[1]
			distance_x = distance_x//2 #lower resolution
			distance_x = distance_x*2

			distance_y = self.center[0]-box_center[0]
			distance_y = distance_y//2 #lower resolution
			distance_y = distance_y*2

			return distance_x, distance_y, angle


		#phi and theta need to be in rads
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

