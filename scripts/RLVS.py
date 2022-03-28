#!/usr/bin/env python3

import roslib
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
from gym.utils import seeding

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
        self.z_initial = 0.0
        self.np_random, seed = seeding.np_random(None)

        # Initialize variables
        self.timestep = 1
        self.current_episode = 1
        self.episodic_reward = 0.0
        self.previous_state = np.zeros(num_states)
        self.action = np.zeros(num_actions)
        self.previous_action = np.zeros(num_actions)
        self.done = False
        self.max_distance = 10.0
        self.max_timesteps = 512
        
        # Define line taken from detector
        self.desired_pos.z = 2
        self.distance, self.angle = 0, 0


    def initial_pose(self):
        action_mavros = AttitudeTarget()
        action_mavros.type_mask = 7
        action_mavros.thrust = 0.5 # Altitude hold
        action_mavros.orientation = self.rpy2quat(0.0,0.0,0.0) # Zero yaw
        self.pub_action.publish(action_mavros)


    # Convert roll, pitch, yaw (in degrees) to quaternion
    def rpy2quat(self,roll,pitch,yaw):
        
        q = Quaternion()
        r = np.deg2rad(roll)
        p = np.deg2rad(pitch)
        y = 0.0

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
        # Initialize to random position
        if self.x_initial == 0.0:
            reset_position =  self.np_random.uniform(low = -0.15, high = 0.15, size=(2,))
            reset_position_z =  self.np_random.uniform(low = -0.15, high =  0.15, size=(1,)) 

            self.x_initial = reset_position[0]
            self.y_initial = reset_position[1]  
            self.z_initial = reset_position_z[0]

        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = self.x_initial
        position_reset.position.y = self.y_initial
        position_reset.position.z = self.z_initial
        position_reset.yaw = 0.0
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


    def PoseCallback(self, msg):

        # Read Current Position
        self.distance, self.angle = Detector()

        exceeded_x_limits = abs(x_position) > self.max_distance
        exceeded_y_limits = abs(y_position) > self.max_distance
        exceeded_z_limits = abs(z_position) > self.max_distance
        exceeded_bounds = exceeded_x_limits or exceeded_y_limits or exceeded_z_limits

        # Check done signal which indicates whether s' is terminal. The episode is terminated when the quadrotor is out of bounds or after a max # of timesteps
        if exceeded_bounds and not self.done : # Bounds around desired position
            print("Exceeded Bounds --> Return to initial position")
            self.done = True 
        elif self.timestep > self.max_timesteps and not self.done:
            print("Reached max number of timesteps --> Return to initial position")   
            self.done = True 

        if self.done:
            # Go to Initial position
            self.go_to_start()
            # When reach the inital position, begin next episode       
            if abs(x_position-self.x_initial)<0.15 and abs(y_position-self.y_initial)<0.15 and abs(z_position-self.z_initial)<0.15:
                self.reset()
                self.x_initial = 0.0
                self.y_initial = 0.0
                self.z_initial = 0.0
                 
                print("Reset")                   
                print("Begin Episode %d" %self.current_episode)                 
        else:           
            # Compute the current state: position error, velocity and roll,pitch
            self.current_state = np.array([self.distance , self.angle , self.desired_pos.z - z_position ,
                                             x_velocity , y_velocity, z_velocity, roll, pitch, yaw np.rad2deg(x_angular), np.rad2deg(y_angular), 
                                             self.previous_action[0], self.previous_action[1], self.previous_action[2]], self.previous_action[3])
            # Compute reward from the 2nd timestep and after
            if self.timestep > 1:

                # Compute Reward: use minus because we want to maximize reward
                position_error = abs(self.current_state[0])+abs(self.current_state[1]) + abs(self.current_state[2])

                # Weight for position error 
                weight_position = 1.6

                # Oscillation suppression -> smooth output action
                delta_roll = abs(self.action[0]-self.current_state[11])/angle_max # normalized -> max movement from previous to current action is 2 (e.g from -10 to 10)
                delta_pitch = abs(self.action[1]-self.current_state[12])/angle_max
                delta_yaw = abs(self.action[2]-self.current_state[13])/angle_max
                delta_zdot = abs(self.action[3]-self.current_state[14])/max_vel_up
                # delta_action = math.sqrt(delta_roll**2 + delta_pitch**2 + delta_zdot**2)/2 # max movement is e.g from [-10,-10] to [10,10] -> delta roll, delta pitch = 20 [2 normalized]
                delta_action = delta_roll + delta_yaw + delta_zdot
                # delta_action = delta_roll**2 + delta_pitch**2 + delta_zdot**2


                weight_smoothness = 0.30

                weight_action = 0.10/max(position_error,0.01)
                action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max + abs(self.action[2])/angle_max + abs(self.action[3])/max_vel_up
                # action = (self.action[0])**2/angle_max + (self.action[1])**2/angle_max + (self.action[2])**2/max_vel_up
                
                self.reward = -weight_position*position_error - weight_smoothness*delta_action
                self.reward += -weight_action*action

                sparse_reward = 0.0

                if abs(self.current_state[0])<abs(self.previous_state[0]):
                    self.reward+= sparse_reward
                if abs(self.current_state[1])<abs(self.previous_state[1]):
                    self.reward+= sparse_reward
                if abs(self.current_state[2])<abs(self.previous_state[2]):
                    self.reward+= sparse_reward

               
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
                    print("Delta action error: ", delta_action)
                    print("Total reward: ",self.reward)
                
            self.previous_action = self.action                  

            # Pick an action according to actor network
            tf_current_state = tf.expand_dims(tf.convert_to_tensor(self.current_state), 0)
            tf_action = tf.squeeze(actor_model(tf_current_state))
            noise = ou_noise()
            self.action = tf_action.numpy() + noise  # Add exploration strategy
            self.action[0] = np.clip(self.action[0], angle_min, angle_max)
            self.action[1] = np.clip(self.action[1], angle_min, angle_max)
            self.action[2] = np.clip(self.action[2], angle_min, angle_max)
            self.action[3] = np.clip(self.action[3], max_vel_down, max_vel_up)

            if self.timestep%100 == 0:
                print("Next action: ", tf_action.numpy())
                print("Noise: ", noise)
                print("Noisy action: ", self.action)

            # Roll, Pitch, Yaw in Degrees
            roll_des = self.action[0]
            pitch_des = self.action[1] 
            yaw_des = self.action[2]

            # Vertical Desired velocity
            zdot_des = self.action[3]

            if zdot_des > 0 :
                thrust = 0.5*zdot_des/2.5 + 0.5
            else:
                thrust = -0.5*zdot_des/max_vel_down + 0.5    

            # Convert to mavros message and publish desired attitude
            action_mavros = AttitudeTarget()
            action_mavros.type_mask = 7
            action_mavros.thrust = thrust #
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
    h1 = layers.Dense(64, activation="tanh")(inputs)
    h2 = layers.Dense(64, activation="tanh")(h1)    
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(h2)

    # Output of tanh is [-1,1] so multiply with the upper control action
    outputs = outputs * [angle_max, angle_max, max_vel_up]
        
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

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
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

    num_actions = 4 # commanded vertical velocity, roll and yaw
    num_states = 11 + num_actions # x,y,z position error, x,y,z velocity, roll and pitch, angular velocity +++++ previous action
    # num_states = 8 # x,y,z position error, x,y,z velocity, roll and pitch

    angle_max = 10.0 
    angle_min = -10.0 # constraints for commanded roll and pitch

    max_vel_up = 1.5 # Real one is 2.5
    max_vel_down = -1.5 # constraints for commanded vertical velocity

    actor_model = get_actor()
    print("Actor Model Summary")
    print(actor_model.summary())

    critic_model = get_critic()
    print("Critic Model Summary")
    print(critic_model.summary())

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Load pretrained weights
    # actor_model.load_weights('/home/fotis/rl_ws/ddpg_actor.h5')
    # critic_model.load_weights('/home/fotis/rl_ws/ddpg_critic.h5')

    # target_actor.load_weights('/home/fotis/rl_ws/ddpg_target_actor.h5')
    # target_critic.load_weights('/home/fotis/rl_ws/ddpg_target_critic.h5')

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    # Define optimizer
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005   

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = [] 
    episodes = []

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

