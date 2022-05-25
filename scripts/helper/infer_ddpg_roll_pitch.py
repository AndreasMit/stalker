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
from BoxToLineClass import line_detector

#-------------------------------- CLASS ENVIRONMENT --------------------------------#

class Environment:

    def __init__(self):
        
        # Publishers
        self.pub_pos = rospy.Publisher("/mavros/setpoint_raw/local",PositionTarget,queue_size=10000)
        self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)
        
        # Initialize yaw to zero
        self.initial_pose()

        # Reset to initial positions
        self.x_initial = 0.0
        self.y_initial = 0.0
        self.z_initial = 7.0
        self.yaw_initial = 90.0

        #initialize current position
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position= 7.0
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.z_velocity = 0.0 
        self.x_angular = 0.0
        self.y_angular = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 90.0

        # define good limits
        self.good_angle = 10
        self.good_distance = 50 
        self.exceeded_bounds = False
        self.to_start = False
        self.done = False

        # Initialize variables
        self.action = np.zeros(num_actions)

        
        # Define Subscriber !edit type
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        
        # Define line taken from detector
        self.box = PREDdata()
        self.desired_pos_z = 5.0
        self.desired_vel_x = 0.5
        self.distance, self.angle = 0, 0
        self.new_pose = False

        self.Line = line_detector()

        self.timestep = 0 


    def initial_pose(self):
        action_mavros = AttitudeTarget()
        action_mavros.type_mask = 7
        action_mavros.thrust = 0.5 # Altitude hold
        action_mavros.orientation = self.rpy2quat(0.0,0.0,90.0) # 90 yaw
        self.pub_action.publish(action_mavros)


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

    def go_to_start(self):
        #go to the last point when you had a good detection
        #that point is stored in x/y/z initial
        # print('going to start')
        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = self.x_initial
        position_reset.position.y = self.y_initial
        position_reset.position.z = self.z_initial
        position_reset.yaw = self.yaw_initial
        self.pub_pos.publish(position_reset) 

    def reset(self):
        self.exceeded_bounds = False  
        self.to_start  = False 
        self.done = False

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
            self.distance , self.angle = self.Line.compute(self.box, self.roll, self.pitch, self.z_position)
            # print(self.distance, self.angle, self.z_position)
            # print(self.angle, self.yaw)
            if self.distance == 10000 and self.angle == 0 :
                self.exceeded_bounds = True
            if abs(self.angle) < 0.5 and abs(self.distance) > 270:
                self.exceeded_bounds = True
            elif abs(self.angle) > 89.5: # this includes being vertical to the pavement but also cases when the detection is on the edge of image and is not reliable
                self.exceeded_bounds = True 
            #no need for new starting position

            if self.exceeded_bounds and not self.done:
                print("Exceeded Bounds --> Return to initial position")
                self.done = True

            if self.exceeded_bounds:
                # instead go to last frame that had detection
                if not self.to_start:
                    self.go_to_start()
                # When reach the inital position, begin next episode    
                if abs(self.x_position-self.x_initial)<0.2 and abs(self.y_position-self.y_initial)<0.2 and abs(self.z_position-self.z_initial)<0.2 :
                    self.to_start = True
                    # print('setting yaw')
                    action_mavros = AttitudeTarget()
                    action_mavros.type_mask = 7
                    action_mavros.thrust = 0.5 # Altitude hold
                    action_mavros.orientation = self.rpy2quat(0.0,0.0,self.yaw_initial) 
                    self.pub_action.publish(action_mavros)
                    if abs(self.yaw - self.yaw_initial)<10 :
                        self.reset()                 
                        print("Reset")                   
                else:
                    self.to_start = False               
            else:           
                # Compute the current state
                max_distance = 360 #pixels
                max_velocity = 2 #m/s
                max_angle = 90 #degrees #bad name of variable ,be careful there is angle_max too for pitch and roll.

                #STATE
                #normalized values only -> [0,1]
                self.current_state = np.array([self.distance/max_distance , min(self.y_velocity/max_velocity, 1), min((self.x_velocity - self.desired_vel_x)/max_velocity, 1) ])
                 
                # Pick an action according to actor network
                tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
                tf_action = tf.squeeze(target_actor(tf_current_state))
                self.action = tf_action.numpy()
                # self.action = self.action[0]
                # print(self.action)
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)

                distances.append(self.distance/max_distance)
                xvels.append(min((self.x_velocity - self.desired_vel_x)/max_velocity, 1))

                if self.timestep % 30 == 0:
                    plt.figure(0)
                    plt.plot(distances, 'b')
                    plt.plot(xvels, 'r')
                    plt.grid()
                    plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/st_co'+str(checkpoint)+'/infer_distance_error')
                    # print(self.z_position)
                    print(self.x_velocity)

                self.timestep += 1
                # Roll, Pitch, Yaw in Degrees
                roll_des = self.action[0]
                pitch_des = self.action[1]
                yaw_des = 90 #self.action[1] + self.yaw  #differences in yaw
                # print(yaw_des)

                # Convert to mavros message and publish desired attitude
                action_mavros = AttitudeTarget()
                action_mavros.type_mask = 7
                action_mavros.thrust = 0.5
                action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,yaw_des)
                self.pub_action.publish(action_mavros)
     

def get_actor():

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    h1 = layers.Dense(128, activation="tanh")(inputs)
    h2 = layers.Dense(128, activation="tanh")(h1)    
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(h2)

    # Output of tanh is [-1,1] so multiply with the upper control action
    outputs = outputs * [angle_max, angle_max]
        
    model = tf.keras.Model(inputs, outputs)

    return model  


if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    tf.compat.v1.enable_eager_execution()

    num_actions = 2
    num_states = 3  

    angle_max = 3.0 
    angle_min = -3.0 # constraints for commanded roll and pitch
    yaw_max = 5.0 #how much yaw should change every time
    yaw_min = -5.0

    checkpoint = 14

    target_actor = get_actor()
    target_actor.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/st_co'+str(checkpoint)+'/ddpg_target_actor.h5')

    distances = []
    angles = []
    xvels = []
    Environment()
    

    r = rospy.Rate(20)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()        

