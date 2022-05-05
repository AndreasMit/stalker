#!/usr/bin/env python3

import rospy
import tensorflow as tf
import numpy as np
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
from boxtoline import Line_detector
from util import *
import var

#-------------------------------- CLASS ENVIRONMENT --------------------------------#

class Environment:

    def __init__(self):
        
        # Publishers
        self.pub_pos = rospy.Publisher("/mavros/setpoint_raw/local",PositionTarget,queue_size=10000)
        self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)
        
        # Initialize yaw
        self.initial_pose()

        # Reset to initial positions
        self.x_initial = 0.0
        self.y_initial = 0.0
        self.z_initial = 5.0
        self.yaw_initial = 90.0

        #initialize current position
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position= 5.0
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

        # Initialize variables
        self.timestep = 1
        self.current_episode = 1
        self.episodic_reward = 0.0
        self.previous_state = np.zeros(num_states)
        self.action = np.zeros(num_actions)
        self.previous_action = np.zeros(num_actions)
        self.done = False
        self.max_timesteps = 512
        
        # Define Subscriber !edit type
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        
        # Define line taken from detector
        self.box = PREDdata()
        self.desired_pos_z = 5.0
        self.desired_vel_x = 1
        self.distance, self.angle = 0, 0
        self.new_pose = False

        self.Line = line_detector()


    def initial_pose(self):
        action_mavros = AttitudeTarget()
        action_mavros.type_mask = 7
        action_mavros.thrust = 0.5 # Altitude hold
        action_mavros.orientation = rpy2quat(0.0,0.0,90.0) # 90 yaw
        self.pub_action.publish(action_mavros)     

    def go_to_start(self):
        #go to the last point when had a good detection
        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = self.x_initial
        position_reset.position.y = self.y_initial
        position_reset.position.z = self.z_initial
        position_reset.yaw = self.yaw_initial
        self.pub_pos.publish(position_reset) 

    def reset(self):
        # If done, the episode has terminated -> save the episode's reward
        ep_reward_list.append(self.episodic_reward*self.max_timesteps/self.timestep)
        # ep_reward_list.append(self.episodic_reward)
        # Mean episodic reward of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        episodes.append(self.current_episode)
        print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward*self.max_timesteps/self.timestep))
        # print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward))
        print("Episode * {} * Avg Reward is ==> {}".format(self.current_episode, avg_reward))
        avg_reward_list.append(avg_reward)
        # Save the weights every 30 episodes to a file
        if self.current_episode % 30 == 0.0:
            actor_model.save_weights("ddpg_actor.h5")
            critic_model.save_weights("ddpg_critic.h5")

            target_actor.save_weights("ddpg_target_actor.h5")
            target_critic.save_weights("ddpg_target_critic.h5")    

            print("-----Weights saved-----") 

            pylab.plot(episodes, ep_reward_list, 'b')
            pylab.plot(episodes, avg_reward_list, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig("DDPG_score.png")
                print("-----Plots saved-----")
            except OSError:
                pass            

        # Reset episodic reward and timestep to zero
        self.episodic_reward = 0.0
        self.current_episode += 1
        self.timestep = 1
        self.done = False
        self.exceeded_bounds = False  
        self.to_start  = False 

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
        self.roll, self.pitch, self.yaw = quat2rpy(quat)
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
            elif abs(self.distance) < self.good_distance and abs(self.angle) < self.good_angle and self.angle!=0:
                # print('good position')
                # print(self.distance, self.angle)
                self.x_initial = self.x_position
                self.y_initial = self.y_position
                # self.z_initial = self.z_position #keep it to 5 meters
                self.yaw_initial = self.yaw

            # Check done signal which indicates whether s' is terminal. The episode is terminated when the quadrotor is out of bounds or after a max # of timesteps
            if self.exceeded_bounds and not self.done : # Bounds around desired position
                print("Exceeded Bounds --> Return to initial position")
                self.done = True 
            elif self.timestep > self.max_timesteps and not self.done:
                print("Reached max number of timesteps --> Return to initial position")   
                self.done = True 

            if self.done:
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
                    action_mavros.orientation = rpy2quat(0.0,0.0,self.yaw_initial) 
                    self.pub_action.publish(action_mavros)
                    if abs(self.yaw - self.yaw_initial)<10 :
                        self.reset()                 
                        print("Reset")                   
                        print("Begin Episode %d" %self.current_episode)  
                else:
                    self.to_start = False               
            else:           
                # Compute the current state
                max_distance = 360 #pixels
                max_velocity = 2 #m/s
                max_angle = 90 #degrees #bad name of variable ,be careful there is angle_max too for pitch and roll.

                #STATE
                #normalized values only -> [0,1]
                self.current_state = np.array([self.distance/max_distance , self.angle/max_angle , self.x_velocity/max_velocity])

                # Compute reward from the 2nd timestep and after
                if self.timestep > 1:

                    #REWARD

                    #penalize big angle and distance from center
                    if self.angle < 2 and abs(self.distance) > 260: # this case is when the box is on the edge of the image and its not realy vertical
                        angle_error = 1
                    else:
                        angle_error = abs(self.angle)/max_angle

                    position_error = abs(self.distance)/max_distance + angle_error
                    weight_position = 50
                    #max 100
                    # print(angle_error, abs(self.distance))

                    #penalize velocity error
                    velocity_error = abs(self.x_velocity - self.desired_vel_x)/max_velocity
                    weight_velocity = 40
                    #max 40

                    # penalize big roll and pitch values
                    #could do it with sqrt
                    action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max + abs(self.action[2])/yaw_max
                    weight_action = 10
                    #max 30

                    #penalize changes in yaw
                    yaw_smooth = abs(self.action[2]-self.previous_action[2])/yaw_max
                    weight_yaw = 30
                    #max 30

                    #total max 200
                    # print(weight_position*position_error, weight_velocity*velocity_error, weight_action*action, weight_yaw*yaw_smooth )
                    # print(self.action[0], self.action[1], self.action[2])
                    # print(self.yaw)
                    #use minus because we want to maximize reward
                    self.reward  = -weight_position*position_error 
                    self.reward += -weight_velocity*velocity_error
                    self.reward += -weight_action*action
                    self.reward += -weight_yaw*yaw_smooth
                    self.reward = self.reward/200 # -> reward is between [-1,0]
                    # dont use the above if you are using 'shaping'
                   
                    # print(self.reward)
                    # Record s,a,r,s'
                    buffer.record((self.previous_state, self.action, self.reward, self.current_state ))

                    self.episodic_reward += self.reward
                    # Optimize the NN weights using gradient descent
                    buffer.learn()
                    # Update the target Networks
                    update_target(target_actor.variables, actor_model.variables, tau)
                    update_target(target_critic.variables, critic_model.variables, tau)  

                    if self.timestep%200 == 0:
                        print("--------------Counter %d--------------" % self.timestep) 
                        print("State: ", self.previous_state)
                        print("Next State: ",self.current_state)
                        print("Previous action: ",self.previous_action)
                        print("Action: ",self.action)
                        print("Position error: ",position_error)
                        print("Total reward: ",self.reward)
                    
                self.previous_action = self.action                  

                # Pick an action according to actor network
                tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
                tf_action = tf.squeeze(actor_model(tf_current_state))
                noise = ou_noise()
                self.action = tf_action.numpy() + noise  # Add exploration strategy
                # print(self.action)
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)
                self.action[2] = np.clip(self.action[2], yaw_min, yaw_max)

                if self.timestep%100 == 0:
                    print("Next action: ", tf_action.numpy())
                    print("Noise: ", noise)
                    print("Noisy action: ", self.action)

                # Roll, Pitch, Yaw in Degrees
                roll_des = self.action[0]
                pitch_des = self.action[1] 
                yaw_des = self.action[2] + self.yaw  #differences in yaw
                # print(yaw_des)

                # Convert to mavros message and publish desired attitude
                action_mavros = AttitudeTarget()
                action_mavros.type_mask = 7
                action_mavros.thrust = 0.5
                action_mavros.orientation = rpy2quat(roll_des,pitch_des,yaw_des)
                self.pub_action.publish(action_mavros)

                self.previous_state = self.current_state
                self.timestep += 1        