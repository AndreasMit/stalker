import numpy as np
from math import *

# Function calling the feature transformation from the image plane on a virtual image plane
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

# print("mp: ", mp)
mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
# print("mp_cartesian: ", mp_cartesian)
mp_cartesian_v = self.featuresTransformation(mp_cartesian, self.phi_imu, self.theta_imu)
# print("mp_cartesian_v: ", mp_cartesian_v)
mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)
# print("mp_pixel_v: ", mp_pixel_v)
