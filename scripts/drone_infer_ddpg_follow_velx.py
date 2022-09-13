#!/usr/bin/env python3

import rospy
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
import math
import time
from tensorflow.keras import layers
from mavros_msgs.msg import PositionTarget
import pylab
from stalker.msg import PREDdata
from BoxToCenter import center_detector
import csv

#-------------------------------- CLASS ENVIRONMENT --------------------------------#

class Environment:

    def __init__(self):
        
        # Publishers
        self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)

        # Reset to initial positions
        self.x_initial = 0.0
        self.y_initial = 0.0
        self.z_initial = 4.0
        self.yaw_initial = 90.0

        #initialize current position
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position= 4.0
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0 
        self.x_angular = 0.0
        self.y_angular = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 90.0

        # define good limits
        self.exceeded_bounds = False
        self.to_start = False
        self.done = False

        # Initialize variables
        self.action = np.zeros(num_actions)
        self.previous_state = np.zeros(num_states)
        self.current_episode = 0
        
        # Define Subscriber !edit type
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        
        # Define line taken from detector
        self.box = PREDdata()
        self.new_pose = False
        self.timestep = 0

        self.detector = center_detector()
        self.distance_x = 0
        self.distance_y = 0
        self.angle = 0
        self.ddist_x = 0
        self.ddist_y = 0
        self.dt = 0
        self.time_prev = 0


    # Convert roll, pitch, yaw (in degrees) to quaternion
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

    def PoseCallback(self,msg):
        self.position = msg
        self.x_position = self.position.pose.pose.position.x
        self.y_position = self.position.pose.pose.position.y
        self.z_position = self.position.pose.pose.position.z
        
        self.x_velocity = self.position.twist.twist.linear.x 
        self.y_velocity = self.position.twist.twist.linear.y
        self.z_velocity = self.position.twist.twist.linear.z 

        self.x_angular = self.position.twist.twist.angular.x
        self.y_angular = self.position.twist.twist.angular.y

        quat = self.position.pose.pose.orientation
        self.roll, self.pitch, self.yaw = self.quat2rpy(quat)
        self.new_pose = True

    def DetectCallback(self, msg):
        #we need updated values for attitude thus
        if self.new_pose == False:
            # print('no new pose')
            return
        else:
            self.new_pose = False
            # Read Current detection
            self.box = msg
            self.distance_x, self.distance_y, self.angle = self.detector.compute(self.box, self.roll, self.pitch, self.z_position)

            #time synching for differantiation
            rostime_now = rospy.get_rostime()
            self.time_now = rostime_now.to_nsec()
            if self.time_prev == 0:
                self.dt = 0
            else:
                self.dt = (self.time_now - self.time_prev)/1e9
            self.time_prev = self.time_now

            if self.distance_x == 10000 and self.distance_y == 10000 :
                self.exceeded_bounds = True

            if self.exceeded_bounds and not self.done:
                print("Exceeded Bounds")         
            else:           
                # Compute the current state
                max_distance_x = 240 #pixels
                max_distance_y = 360
                max_velocity = 2 #m/s
                max_angle = 90 #degrees #bad name of variable ,be careful there is angle_max too for pitch and roll.
                max_derivative = 100

                #STATE
                if self.dt == 0:
                    self.ddist_x = 0
                    self.ddist_y = 0
                else:
                    self.ddist_x = ( self.distance_x - int(self.previous_state[0]*max_distance_x) ) / self.dt
                    self.ddist_y = ( self.distance_y - int(self.previous_state[1]*max_distance_y) ) / self.dt
                    # values -> 2,4,6 pixels (because of resolution reduction in BoxToCenter)
                    # most common 2 pixels movement , /0.1 === *10 => 20 is the most common value 
               
                #normalized values only -> [0,1]
                self.current_state = np.array([self.distance_x/max_distance_x, self.distance_y/max_distance_y, np.clip(self.ddist_x/max_derivative,-1, 1), np.clip(self.ddist_y/max_derivative,-1, 1), np.clip(self.y_velocity/max_velocity, -1, 1), np.clip(self.x_velocity/max_velocity, -1, 1)])
                 
                # Pick an action according to actor network
                tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
                tf_action = tf.squeeze(target_actor(tf_current_state))
                self.action = tf_action.numpy()
                # print(self.action)
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)
                self.action[2] = np.clip(self.action[2], yaw_min, yaw_max)

                with open('src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/logfile'+str(nntry)+'.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    data = [ rospy.get_rostime(), self.distance_x/max_distance_x,self.distance_y/max_distance_y, self.angle/max_angle, self.x_velocity, self.y_velocity, self.z_position , self.action[0], self.action[1], self.action[2] ]
                    writer.writerow(data)

                distances_x.append(self.distance_x/max_distance_x)
                distances_y.append(self.distance_y/max_distance_y)
                angles.append(self.angle/max_angle)
                if self.timestep % 30 == 0:
                    plt.figure(0)
                    plt.title('distance and angle error', fontsize=10)
                    plt.ylim(-0.6,0.6)
                    plt.plot(angles, 'g', label='angle')
                    plt.plot(distances_x, 'b', label='distance_x')
                    plt.plot(distances_y, 'r', label='distance_y')
                    plt.grid()
                    plt.legend()
                    plt.savefig('src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/infer_distance_error'+str(nntry))
                    plt.clf()
                    # print('height: ', self.z_position,', velocity: ' ,self.x_velocity)

                self.timestep += 1
                self.previous_state = self.current_state


                # Roll, Pitch, Yaw in Degrees
                roll_des = self.action[0]
                pitch_des = self.action[1] 
                yaw_des = self.action[2] + self.yaw  #differences in yaw
                # print(yaw_des)

                # Convert to mavros message and publish desired attitude
                action_mavros = AttitudeTarget()
                action_mavros.type_mask = 7
                action_mavros.thrust = 0.5
                action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,yaw_des)
                action_mavros.header.stamp = rospy.get_rostime()
                self.pub_action.publish(action_mavros)
     

def get_actor():

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    h1 = layers.Dense(256, activation="tanh")(inputs)
    h2 = layers.Dense(256, activation="tanh")(h1)    
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(h2)

    # Output of tanh is [-1,1] so multiply with the upper control action
    outputs = outputs * [angle_max, angle_max, yaw_max]
        
    model = tf.keras.Model(inputs, outputs)

    return model  


if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    tf.compat.v1.enable_eager_execution()

    num_actions = 3
    num_states = 6 

    angle_max = 3.0 
    angle_min = -3.0 # constraints for commanded roll and pitch
    yaw_max = 10.0 #how much yaw should change every time
    yaw_min = -10.0

    checkpoint = 3 #checkpoint try
    ntry = 4
    nntry = 3
    target_actor = get_actor()
    target_actor.load_weights('src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_target_actor3.h5')

    distances_x = []
    distances_y = []
    angles = []
    Environment()

    # r = rospy.Rate(20)
    r = rospy.Rate(10)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()        

