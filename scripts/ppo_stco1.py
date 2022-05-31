#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import time
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import PositionTarget
import math

from stalker.msg import PREDdata
from BoxToLineClass import line_detector

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))



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

        # Initialize variables
        self.timestep = 1
        self.current_episode = 0
        self.episodic_reward = 0.0
        self.observation = np.zeros(observation_dimensions)
        self.action = np.zeros(num_actions)
        self.previous_action = np.zeros(num_actions)
        self.done = False
        self.max_timesteps = 1024
        self.ngraph = 0

        self.sum_return = 0.0 
        self.sum_timesteps = 0 #sum_length
        self.steps_per_epoch = 4000
        
        # Define Subscriber !edit type
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.DetectCallback)
        self.sub_position = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.PoseCallback)
        
        # Define line taken from detector
        self.box = PREDdata()
        self.desired_pos_z = 7.0
        self.desired_vel_x = 1
        self.distance, self.angle = 0, 0
        self.new_pose = False

        self.Line = line_detector()


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

    def epoch_done(self):
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = self.buffer.get()

        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {self.sum_return / self.current_episode}. Mean Length: {self.sum_timesteps / self.current_episode}"
        )
        print('episodes of this epoch ', self.current_episode)
        actor.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co"+str(checkpoint)+"/try"+str(ntry)+"/ppo_actor.h5")
        critic.save_weights("/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co"+str(checkpoint)+"/try"+str(ntry)+"/ppo_critic.h5")
        print("-----Weights saved-----")

        plt.plot(self.sum_return / self.current_episode, 'b')
        plt.ylabel('Score')
        plt.xlabel('Steps')
        plt.grid()
        plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co'+str(checkpoint)+'/try'+str(ntry)+'/ppo_score')
        plt.figure()
        plt.plot(distances, 'b')
        plt.plot(angles, 'r')
        plt.grid()
        plt.savefig('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co'+str(checkpoint)+'/try'+str(ntry)+'/distance_and_angle')
        print("-----Plots saved-----")

        self.sum_return = 0.0
        self.current_episode = 0
        self.sum_timesteps = 0

    def reset(self):
        # If done, the episode has terminated -> save the episode's reward
        ep_reward_list.append(self.episodic_reward)
        # Mean episodic reward of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        episodes.append(self.current_episode)
        print("Episode * {} * Cur Reward is ==> {}".format(self.current_episode,self.episodic_reward))
        print("Episode * {} * Avg Reward is ==> {}".format(self.current_episode, avg_reward))
        print("timesteps of this episode ", self.timestep)
        avg_reward_list.append(avg_reward)

        if (self.sum_timesteps > self.steps_per_epoch):
            last_value = critic(self.observation)
        else: #exceeded bounds
            last_value = -1

        buffer.finish_trajectory(last_value) 
        self.sum_return += self.episodic_reward 

        # Reset episodic reward and timestep to zero
        self.episodic_reward = 0.0
        self.current_episode += 1
        self.timestep = 1
        self.done = False
        self.exceeded_bounds = False  
        self.to_start  = False 

        if (self.sum_timesteps> self.steps_per_epoch):
            self.epoch_done()

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
            elif abs(self.angle) < 0.5 and abs(self.distance) > 270: # this case is when the box is on the edge of the image and its not really vertical
                self.exceeded_bounds = True
            elif abs(self.angle) > 89.5: # this includes being vertical to the pavement but also cases when the detection is on the edge of image and is not reliable
                self.exceeded_bounds = True 

            elif abs(self.distance) < self.good_distance and abs(self.angle) < self.good_angle and self.angle!=0:
                # print('good position')
                # print(self.distance, self.angle)
                self.x_initial = self.x_position
                self.y_initial = self.y_position
                # self.z_initial = self.z_position #keep it to 7 meters
                self.yaw_initial = self.yaw

            # Check done signal which indicates whether s' is terminal. The episode is terminated when the quadrotor is out of bounds or after a max # of timesteps
            if self.exceeded_bounds and not self.done : # Bounds around desired position
                print("Exceeded Bounds --> Return to initial position")
                self.done = True 
            elif self.sum_timesteps > self.steps_per_epoch and not self.done:
                print("Reached max number of timesteps --> Return to initial position")   
                self.done = True 

            if self.done:
                if self.timestep < 10: #for some reason we have a false detection of good position
                    self.x_initial = 0.0
                    self.y_initial = 0.0
                    self.z_initial = 7.0
                    self.yaw_initial = 90.0

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
                self.observation_new = np.array([self.distance/max_distance, np.clip(self.y_velocity/max_velocity, -1, 1), self.angle/max_angle , np.clip((self.x_velocity - self.desired_vel_x)/max_velocity,-1 , 1)])

                # Compute reward from the 2nd timestep and after
                if self.timestep > 1:

                    #REWARD

                    #penalize big angle and distance from center
                    angle_error = abs(self.angle)/max_angle
                    distance_error = abs(self.distance)/max_distance
                    position_error = distance_error + angle_error
                    weight_position = 100
                    #max 100
                    # print(angle_error, abs(self.distance))

                    #penalize velocity error
                    velocity_error = min(abs(self.y_velocity)/max_velocity, 1) + min( abs(self.x_velocity - self.desired_vel_x)/max_velocity, 1)
                    weight_velocity = 60
                    #max 40

                    # penalize big roll and pitch values
                    #could do it with sqrt
                    action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max 
                    weight_action = 10
                    #max 30

                    #penalize changes in yaw
                    yaw_smooth = abs(self.action[2])/yaw_max
                    weight_yaw = 10
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
                    self.reward = self.reward/350 # -> reward is between [-1,0]
                    # dont use the above if you are using 'shaping'
                   
                    # print(self.reward)
                    self.episodic_reward += self.reward

                    # Store obs, act, rew, v_t, logp_pi_t
                    buffer.store(self.observation, self.action, self.reward, self.value_t, self.logprobability_t)
                    angles.append(self.angle/max_angle)
                    distances.append(self.distance/max_distance)

                tf_observation = tf.expand_dims(tf.convert_to_tensor(self.observation_new), 0)
                tf_action = tf.squeeze(actor(tf_observation))
                # print(tf.shape(tf_observation))
                # self.logits, tf_action =  sample_action( tf_observation )
                self.action = tf_action.numpy() 
                print(self.action)
                self.action[0] = np.clip(self.action[0], angle_min, angle_max)
                self.action[1] = np.clip(self.action[1], angle_min, angle_max)
                self.action[2] = np.clip(self.action[2], yaw_min, yaw_max)

                self.value_t = critic(tf_observation)
                # self.logprobability_t = logprobabilities(self.logits, self.action)
                self.logprobability_t = 0
                self.previous_action = self.action                  

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

                self.observation = self.observation_new
                self.timestep += 1 
                self.sum_timesteps += 1     


def get_actor():

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(observation_dimensions,))
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
    state_input = layers.Input(shape=(observation_dimensions))
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
    tf.compat.v1.enable_eager_execution()

    # Hyperparameters of the PPO algorithm
    steps_per_epoch = 4000
    clip_ratio = 0.2
    policy_learning_rate = 3e-4
    value_function_learning_rate = 1e-3
    train_policy_iterations = 80
    train_value_iterations = 80
    lam = 0.97
    gamma = 0.99
    target_kl = 0.01

    observation_dimensions = 4
    num_actions = 3

    angle_max = 3.0 
    angle_min = -3.0 # constraints for commanded roll and pitch
    yaw_max = 5.0 #how much yaw should change every time
    yaw_min = -5.0

    # Initialize the buffer
    buffer = Buffer(observation_dimensions, steps_per_epoch)
    
    checkpoint = 0 #checkpoint try
    ntry = 0

    actor = get_actor()
    print("Actor Model Summary")
    print(actor.summary())

    critic = get_critic()
    print("Critic Model Summary")
    print(critic.summary())

    actor.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co'+str(checkpoint)+'/try'+str(ntry)+'/ppo_actor.h5')
    critic.load_weights('/home/andreas/andreas/catkin_ws/src/stalker/scripts/checkpoints/ppo/st_co'+str(checkpoint)+'/try'+str(ntry)+'/ppo_critic.h5')

    critic_optimizer = tf.keras.optimizers.Adam(value_function_learning_rate)
    actor_optimizer = tf.keras.optimizers.Adam(policy_learning_rate)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = [] 
    episodes = []
    distances = []
    angles = []
    rewards = []

    Environment()
    r = rospy.Rate(20)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()    




