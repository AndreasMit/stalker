#!/usr/bin/env python
from __future__ import print_function
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import *
from numpy.linalg import norm
from sensor_msgs.msg import Image, Imu


self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)

# Callback function updating the IMU measurements (rostopic /mavros/imu/data)


def updateImu(self, msg):
    self.phi_imu = msg.orientation.x
    self.theta_imu = msg.orientation.y
    self.psi_imu = msg.orientation.z
    self.w_imu = msg.orientation.w
    self.phi_imu, self.theta_imu, self.psi_imu = euler_from_quaternion(
        [self.phi_imu, self.theta_imu, self.psi_imu, self.w_imu])


# Function calling the feature transformation from the image plane on a virtual image plane


    def featuresTransformation(self, mp, phi, theta):

        Rphi = np.array([[1.0, 0.0, 0.0], [0.0, cos(phi), -sin(phi)],
                        [0.0, sin(phi), cos(phi)]]).reshape(3, 3)
        Rtheta = np.array([[cos(theta), 0.0, sin(theta)], [
                          0.0, 1.0, 0.0], [-sin(theta), 0.0, cos(theta)]]).reshape(3, 3)
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

        mp_cartesian = np.array(
            [X_0, Y_0, Z_0, X_1, Y_1, Z_1, X_2, Y_2, Z_2, X_3, Y_3, Z_3])

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

# print("mp: ", mp)
mp_cartesian = self.cartesian_from_pixel(
            mp, self.cu, self.cv, self.ax, self.ay)

mp_cartesian_v = self.featuresTransformation(
            mp_cartesian, self.phi_imu, self.theta_imu)

mp_pixel_v = self.pixels_from_cartesian(
            mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)


mp_des = np.array([420, 472+self.a*self.t, self.z, 367, 483+self.a*self.t,
                           self.z, 327, 2+self.a*self.t, self.z, 377, 0+self.a*self.t, self.z])


cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 1)
cv2.line(cv_image, (int(x_min), 54),
                 (int(x_min), 74), (255, 0, 0), 1)

R_y = np.array([[cos(self.theta_cam), 0.0, sin(self.theta_cam)],
                        [0.0, 1.0, 0.0],
                        [-sin(self.theta_cam), 0.0, cos(self.theta_cam)]]).reshape(3, 3)
sst = np.array([[0.0, -self.transCam[2], self.transCam[1]],
                [self.transCam[2], 0.0, -self.transCam[0]],
                [-self.transCam[1], self.transCam[0], 0.0]]).reshape(3, 3)
T = np.zeros((6, 6), dtype=float)
T[0:3, 0:3] = R_y
T[3:6, 3:6] = R_y
T[0:3, 3:6] = np.dot(sst, R_y)
# print("From body to camera transformation: ", T)
# Interaction matrix, error of pixels and velocity commands calculation (a.k.a control execution)
Lm, er_pix = self.calculateIM(
    mp_pixel_v, mp_des, self.cu, self.cv, self.ax, self.ay)  # TRANSFORM FEATURES


# ......................................
# ......................................
# ......................................

UVScmd = self.quadrotorVSControl_tracking(Lm, er_pix, wave_estimation_final)
# UVScmd = np.dot(T, UVScmd)
UVScmd = np.dot(np.linalg.inv(T), UVScmd)
self.er_pix_prev = er_pix
# print("er_pix_prev: ", self.er_pix_prev)
# print("UVScmd: ", UVScmd)

self.uav_vel_body[0] = UVScmd[0]
self.uav_vel_body[1] = UVScmd[1]
self.uav_vel_body[2] = UVScmd[2]
self.uav_vel_body[3] = UVScmd[5]
