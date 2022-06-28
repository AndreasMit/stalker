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
from stalker.msg import PREDdata
from BoxToCenter import center_detector

#-------------------------------- NOISE CLASS --------------------------------#


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

#-------------------------------- CLASS BUFFER --------------------------------#

class Buffer:
    def __init__(self, buffer_capacity = 100000, batch_size = 64):

        #Number of experiences to store at max
        self.buffer_capacity = buffer_capacity
        
        #Number of tuples to train on
        self.batch_size = batch_size

        #Number of times record() was called
        self.buffer_counter = 0

        #We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    #Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):  

        #Set index to zeros if buffer_capacity is exceeded and replace old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1  

    def learn(self):

        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.

        with tf.GradientTape() as tape:

            target_actions = target_actor(next_state_batch, training=True)
            #Compute the real expected return
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True) 
            #Define the output of the critic according to the current batch
            critic_value = critic_model([state_batch, action_batch], training=True)
            #Define the Loss Function (real expected return - output of critic)**2
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        #Do gradient ascent to the critic model according to the loss
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given (we want gradient ascent, but by default gradient descent is used)
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))  

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
        self.z_initial = 4.0
        self.yaw_initial = 90.0

        self.x_initial_noise = np.random.uniform(-2, 2)
        self.y_initial_noise = np.random.uniform(-2, 2)

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
        self.max_timesteps = 1024
        self.ngraph = 0
        self.max_avg_reward = -1000
        
        # Define Subscriber
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        self.sub_target = rospy.Subscriber("/robot/robotnik_base_control/odom", Odometry, self.TargetCallback)
        
        # Define line taken from detector
        self.box = PREDdata()
        self.desired_pos_z = 4.0
        self.new_pose = False

        self.detector = center_detector()
        self.distance_x = 0
        self.distance_y = 0
        self.angle = 0
        self.ddist_x = 0
        self.ddist_y = 0
        self.dt = 0
        self.time_prev = 0


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
        # go the target, given its coordinates
        # that point is stored in x/y/z initial
        # print('going to target')
        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = self.x_initial + self.x_initial_noise
        position_reset.position.y = self.y_initial + self.y_initial_noise
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
        print("timesteps :", self.timestep)
        print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward*self.max_timesteps/self.timestep))
        # print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward))
        print("Episode * {} * Avg Reward is ==> {}".format(self.current_episode, avg_reward))
        avg_reward_list.append(avg_reward)
       
        if (avg_reward > self.max_avg_reward and avg_reward != 0):
            self.max_avg_reward = avg_reward
            actor_model.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow"+str(checkpoint)+"/try"+str(ntry)+"/ddpg_actor"+str(self.ngraph)+".h5")
            critic_model.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow"+str(checkpoint)+"/try"+str(ntry)+"/ddpg_critic"+str(self.ngraph)+".h5")
            target_actor.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow"+str(checkpoint)+"/try"+str(ntry)+"/ddpg_target_actor"+str(self.ngraph)+".h5")
            target_critic.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow"+str(checkpoint)+"/try"+str(ntry)+"/ddpg_target_critic"+str(self.ngraph)+".h5")    
            print("-----Weights saved-----")

        # Save the weights every 30 episodes to a file
        if self.current_episode % 10 == 0.0:
            plt.figure() 
            plt.plot(ep_reward_list, 'b', label='ep_reward')
            plt.plot(avg_reward_list, 'r', label='avg_reward')
            plt.ylabel('Score')
            plt.xlabel('Episodes')
            plt.grid()
            plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_score'+str(self.ngraph))
            print("-----Plots saved-----")
            # plt.figure(1)
            # plt.scatter(distances, angles, c=rewards)
            # plt.grid()
            # plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/reward_per_error'+str(ntry)+'')
            plt.figure()
            plt.plot(angles, 'g', label='angle')
            plt.plot(distances_x, 'b',label='distance_x')
            plt.plot(distances_y, 'r',label='distance_y')
            plt.grid()
            plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/distancexy'+str(self.ngraph))

        if self.current_episode % 200 == 0.0:
            self.ngraph += 1
            #we do this so we reduce memory used and take less time to save the graphs (less delay in training)
            ep_reward_list.clear()
            avg_reward_list.clear()
            distances_x.clear()
            distances_y.clear()
            angles.clear()
            self.max_avg_reward = -1000 #reset for every 200 episodes, we get the max weights in each graph


        # Reset episodic reward and timestep to zero
        self.episodic_reward = 0.0
        self.current_episode += 1
        self.timestep = 1
        self.done = False
        self.exceeded_bounds = False  
        self.to_start  = False 
        # random init again for each episode
        self.x_initial_noise = np.random.uniform(-2, 2)
        self.y_initial_noise = np.random.uniform(-2, 2)

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

    def TargetCallback(self,msg):
        self.x_initial = (-1.0)*msg.pose.pose.position.y  
        self.y_initial = msg.pose.pose.position.x + 2.0   #initial relative position of drone, odom doesnt use global coordinates!
        # the above signs and additions are needed to transform the frames coords

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
            
            # print(self.distance, self.angle, self.z_position)
            # print(self.angle, self.yaw)
            if self.distance_x == 10000 and self.distance_y == 10000 :
                self.exceeded_bounds = True

            # Check done signal which indicates whether s' is terminal. The episode is terminated when the quadrotor is out of bounds or after a max # of timesteps
            if self.exceeded_bounds and not self.done : # Bounds around desired position
                print("Exceeded Bounds --> Return to initial position")
                self.done = True 
            elif self.timestep > self.max_timesteps and not self.done:
                print("Reached max number of timesteps --> Return to initial position") 
                self.reward += 100  
                self.reset()                 
                print("Reset")                   
                print("Begin Episode %d" %self.current_episode)
                #we miss 2 timesteps between episodes while bot is still moving
                
            #only when exceeding bounds we do the following
            if self.done:
                self.go_to_start()
                if abs(self.x_position-self.x_initial-self.x_initial_noise)<0.3 and abs(self.y_position-self.y_initial-self.y_initial_noise)<0.3 and abs(self.z_position-self.z_initial)<0.1 :
                    self.reset()                 
                    print("Reset")                   
                    print("Begin Episode %d" %self.current_episode)      

            else:           
                # Compute the current state
                max_distance_x = 240 #pixels
                max_distance_y = 360
                max_velocity = 2 #m/s
                max_angle = 90 #degrees #bad name of variable ,be careful there is angle_max too for pitch and roll.
                max_derivative = 100

                #STATE
                #dt is around 0.1 as the mavros topic runs at 10Hz
                if self.dt == 0:
                    self.ddist_x = 0
                    self.ddist_y = 0
                else:
                    self.ddist_x = ( self.distance_x - int(self.previous_state[0]*max_distance_x) ) / self.dt
                    self.ddist_y = ( self.distance_y - int(self.previous_state[1]*max_distance_y) ) / self.dt
                    # values -> 2,4,6 pixels (because of resolution reduction in BoxToCenter)
                    # most common 2 pixels movement , /0.1 === *10 => 20 is the most common value 
                # ! bad reslution
                
                # print(self.angle)
                # print(min(self.ddist_y/max_derivative, 1))
                #normalized values only -> [0,1]
                self.current_state = np.array([self.distance_x/max_distance_x, self.distance_y/max_distance_y, self.angle/max_angle, np.clip(self.ddist_x/max_derivative,-1, 1), np.clip(self.ddist_y/max_derivative,-1, 1), np.clip(self.y_velocity/max_velocity, -1, 1), np.clip(self.x_velocity/max_velocity, -1, 1)])


                # Compute reward from the 2nd timestep and after
                if self.timestep > 1:

                    #REWARD
                    angle_error = abs(self.angle)/max_angle
                    distance_error = abs(self.distance_x)/max_distance_x + abs(self.distance_y)/max_distance_y 
                    position_error = distance_error + 0.5*angle_error 
                    weight_position = 100
                    #max 200

                    #penalize derivative error
                    velocity_error = min(abs(self.ddist_x/max_derivative),1) + min(abs(self.ddist_y/max_derivative),1)
                    weight_velocity = 30

                    # penalize big roll and pitch values
                    #could do it with sqrt
                    action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max + abs(self.action[2])/yaw_max
                    weight_action = 10

                    #use minus because we want to maximize reward
                    self.reward  = -weight_position*position_error 
                    self.reward += -weight_velocity*velocity_error
                    self.reward += -weight_action*action
                    self.reward = self.reward/340 # -> reward is between [-1,0]
                    
                    # Record s,a,r,s'
                    buffer.record((self.previous_state, self.action, self.reward, self.current_state ))

                    self.episodic_reward += self.reward
                    # Optimize the NN weights using gradient descent
                    buffer.learn()
                    # Update the target Networks
                    update_target(target_actor.variables, actor_model.variables, tau)
                    update_target(target_critic.variables, critic_model.variables, tau) 

                    distances_x.append(self.distance_x/max_distance_x)
                    distances_y.append(self.distance_y/max_distance_y)
                    angles.append(self.angle/max_angle) 

                    
                self.previous_action = self.action                  

                # Pick an action according to actor network
                tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
                tf_action = tf.squeeze(actor_model(tf_current_state))
                noise = ou_noise()
                self.action = tf_action.numpy() + noise  # Add exploration strategy
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)
                self.action[2] = np.clip(self.action[2], yaw_min, yaw_max)

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
                self.pub_action.publish(action_mavros)
                
            
                self.previous_state = self.current_state
                self.timestep += 1        

#-------------------------------- MAIN --------------------------------#


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau)) 


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

def get_critic():

    # The critic NN has 2 inputs: the states and the actions. Use 2 seperate NN and then concatenate them
    # State as input
    state_input = layers.Input(shape=(num_states))
    h1_state = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(h1_state)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model 

if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    
    # With eager execution, operations are executed as they are 
    # defined and Tensor objects hold concrete values, which 
    # can be accessed as numpy.ndarray`s through the numpy() method.
    tf.compat.v1.enable_eager_execution()

    num_actions = 3 
    num_states = 7

    angle_max = 3.0 
    angle_min = -3.0 # constraints for commanded roll and pitch
    yaw_max = 5.0 #how much yaw should change every time
    yaw_min = -5.0


    checkpoint = 1 #checkpoint try
    ntry = 1

    actor_model = get_actor()
    # print("Actor Model Summary")
    # print(actor_model.summary())

    critic_model = get_critic()
    # print("Critic Model Summary")
    # print(critic_model.summary())

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Load pretrained weights
    # actor_model.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_actor.h5')
    # critic_model.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_critic.h5')

    # target_actor.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_target_actor.h5')
    # target_critic.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/follow'+str(checkpoint)+'/try'+str(ntry)+'/ddpg_target_critic.h5')

    # Learning rate for actor-critic models
    critic_lr = 0.001
    actor_lr = 0.0001

    # Define optimizer
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.001   

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = [] 
    episodes = []

    distances_x = []
    distances_y = []
    angles = []
    rewards = []

    Environment()

    # buffer = Buffer(100000, 1000)
    buffer = Buffer(100000, 64)

    std_dev = 0.1
    # std_dev = 0.2

    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

    r = rospy.Rate(20)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()        

