#!/usr/bin/env python
import rospy
import roslib
import tensorflow as tf 
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import PositionTarget
import pylab
import copy

class Actor_Model:
    # num_states, num_actions: int, max_action, lr(learning rate): float and optimizer: instance of tf.keras.optimizers
    def __init__(self, num_states, num_actions, max_action, lr, optimizer):
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_action = max_action

        # NN with 2 hidden layers with 64 neurons each and tanh activation function everywhere (maybe try linear activation function for the output layer)
        inputs = Input(shape = (self.num_states,))
        h1 = Dense(64, activation = "tanh", kernel_initializer = tf.random_normal_initializer(stddev = 0.01))(inputs)
        h2 = Dense(64, activation = "tanh", kernel_initializer = tf.random_normal_initializer(stddev = 0.01))(h1)
        outputs = Dense(self.num_actions, activation = "tanh")(h2)

        # Tanh function: R -> [-1,1], so multiply with the max legal action
        outputs = outputs*self.max_action
        
        # Define the actor model 
        self.Actor = Model(inputs = inputs, outputs = outputs)

        # Configure the model for training, where loss is any objective function like loss = fn(y_true,y_pred) (ground truth and predicted values)
        self.Actor.compile(loss = self.ppo_loss, optimizer = optimizer(lr=lr))
        
        print(self.Actor.summary())

    # Define Loss Function y_true = np.array([[...],[...],[...]])
    def ppo_loss(self, y_true, y_pred):
        advantages, actions, logp_old = y_true[:,:1], y_true[:, 1 : 1 + self.num_actions], y_true[:, 1 + self.num_actions]  # y_true[:,:1] differs with y_true[:,0]
        
        # ε for clipping
        epsilon = 0.2

        logp = self.gaussian_likelihood(actions, y_pred)
        ratio = K.exp(logp - logp_old)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1 + epsilon)*advantages, (1 - epsilon)* advantages)

        actor_loss = -K.mean(K.minimum(p1,p2)) # K.mean computes mean of a tensor alongside a specific axis, K.minimum: element-wise minimum of two tensors
        # actor_loss = -K.mean(p1) # K.mean computes mean of a tensor alongside a specific axis, K.minimum: element-wise minimum of two tensors

        return actor_loss

    def gaussian_likelihood(self, actions, pred):
        log_std = -0.5* np.ones(self.num_actions, dtype = np.float32)*4
        pre_sum = -0.5*(((actions-pred)/(K.exp(log_std)+1e-8))**2 +2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis = 1)    

    def predict(self, state):
        return self.Actor.predict(state)  


class Critic_Model:
    def __init__(self, num_states, lr, optimizer):
        self.num_states = num_states
        inputs = Input(shape = (self.num_states,))

        h1 = Dense(64, activation = "relu", kernel_initializer = tf.random_normal_initializer(stddev = 0.01))(inputs)
        h2 = Dense(64, activation = "relu", kernel_initializer = tf.random_normal_initializer(stddev = 0.01))(h1)
        value = Dense(1, activation = None)(h2)

        self.Critic = Model(inputs = inputs, outputs = value)
        self.Critic.compile(loss = self.critic_loss, optimizer = optimizer(lr=lr))

        print(self.Critic.summary())

    def critic_loss(self, y_true, y_pred):
        value_loss = 0.5*K.mean((y_true-y_pred)**2)
        return value_loss

    def predict(self,state):
        return self.Critic.predict(state)   


class PPOAgent: 
    def __init__(self):
        # States: position error, velocity error, roll and pitch error
        self.num_states = 6
        # Actions: commanded Roll and Pitch
        self.num_actions = 2  
        self.max_action = 1.0

        self.lr = 0.00025 # Learning Rate
        # self.lr = 0.001 # Learning Rate

        self.optimizer = tf.keras.optimizers.Adam

        # Create Actor - Critic Network models
        self.Actor = Actor_Model(num_states = self.num_states, num_actions = self.num_actions, max_action = self.max_action, lr = self.lr, optimizer= self.optimizer)
        self.Critic = Critic_Model(num_states = self.num_states, lr = self.lr, optimizer = self.optimizer)

        # Save Weights
        self.Actor_name = "PPO_Actor.h5"
        self.Critic_name = "PPO_Critic.h5"

        # Publishers
        self.pub_pos = rospy.Publisher("/mavros/setpoint_raw/local",PositionTarget,queue_size=10000)
        self.pub_action = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10000)

        self.initial_pose()

        # Define target position
        self.desired_pos = Point()
        self.desired_pos.x = 2.0
        self.desired_pos.y = 2.0

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots        

        # Space Bounds (episode termination)
        self.max_radius = 5.0

        # Initialize Variables
        self.state = np.ones(self.num_states, dtype=np.float32)
        self.episodic_reward = 0.0
        self.episode = 1.0
        self.done = False # Episode terminated
        self.timestep = 1
        self.Training_batch = 512
        self.states, self.next_states, self.actions, self.rewards, self.dones, self.logp_ts = [], [], [], [], [], []
        self.shuffle = True
        self.epochs = 10 # training epochs 

        # Noise in actions    
        self.log_std = -0.5 * np.ones(self.num_actions, dtype=np.float32)*4
        self.std = np.exp(self.log_std)
        print("Log Standard Deviation: ",self.log_std)
        print("Standard Deviation: ",self.std)

        # Define Subscriber
        self.sub_cur_pos = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.run_batch)         


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

        # roll = np.rad2deg(roll)
        # pitch = np.rad2deg(pitch)
        # yaw = np.rad2deg(yaw)  

        return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)  

    def run_batch(self, msg):   

        # Read next Pose
        self.next_pos = msg
        x_position = self.next_pos.pose.pose.position.x
        y_position = self.next_pos.pose.pose.position.y
        z_position = self.next_pos.pose.pose.position.z

        x_velocity = self.next_pos.twist.twist.linear.x 
        y_velocity = self.next_pos.twist.twist.linear.y 

        quat = self.next_pos.pose.pose.orientation
        roll, pitch, yaw = self.quat2rpy(quat)

        # Check done signal which indicates whether s' is terminal. The episode is terminated when the quadrotor is out of bounds or after a max # of timesteps
        if (abs(x_position) > self.max_radius or abs(y_position) > self.max_radius) and not self.done :
            print("Exceeded Bounds --> Return to initial position")
            self.done = True 
        elif self.timestep > self.Training_batch:
            print("Reached max number of timesteps --> Return to initial position")   
            self.done = True 

        if self.done:
            # Go to Initial position
            self.go_to_start()
            # When reach the inital position, begin next episode       
            if abs(x_position)<0.1 and abs(y_position)<0.1 and abs(z_position-2.0)<0.1:
                # Reset has finished
                self.done = False
                print("Reset")  
                # Do Training
                print("Begin Training")
                self.replay(self.states, self.actions, self.rewards, self.dones, self.next_states, self.logp_ts) 
                # Plot Model
                self.PlotModel(self.episode,self.episodic_reward)
                # Initialize variables again
                self.states, self.next_states, self.actions, self.rewards, self.dones, self.logp_ts = [], [], [], [], [], [] 
                self.episode += 1
                # Reset episodic reward and timestep to zero
                self.episodic_reward = 0.0
                self.timestep = 1                   
                print("Begin Episode %d" %self.episode)                                       

        else: 
            # Compute the next state: position error, velocity and roll,pitch
            self.next_state = np.array([x_position-self.desired_pos.x , y_position-self.desired_pos.y , x_velocity , y_velocity, roll, pitch])
            
            if self.timestep > 1:
                self.reward = self.compute_reward(self.state)
                # Add a constant reward when the quadrotor stays close to the setpoint
                if (self.state[0]**2 + self.state[1]**2 <  1**2):
                    self.reward += 1               

                # Store Variables
                self.states.append(self.state)
                self.next_states.append(self.state)
                self.rewards.append(self.reward)
                self.logp_ts.append(self.logp_t[0])
                self.dones.append(self.done)
                self.actions.append(self.action)

                # print("--------------Counter %d--------------" % self.timestep)  
                # print("Current position x: ",x_position, "  y:", y_position, " z: ", z_position) 
                # print("Current Yaw: ", yaw)
                # print("state: ", self.state)
                # print("Roll: ", roll)
                # print("Pitch: ", pitch)
                # print("Reward : ",self.reward)
                # print("Next State: ",self.next_state)

                # Compute Episodic Reward
                self.episodic_reward += self.reward             

            self.action, self.logp_t = self.act(self.next_state)
            # Publish Action to Gazebo
            self.step(self.action)

            self.state = self.next_state
            self.timestep += 1

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states) # From list of arrays to [[..,..,..],[..,..,..],[..,..,..]....]
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute Advantages using GAE
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        y_true = np.hstack([advantages, actions, logp_ts])

        # Train Actor and Critic Networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=self.shuffle) 

        print("Actor Loss: ", np.sum(a_loss.history['loss']))      
        print("Critic Loss: ", np.sum(c_loss.history['loss']))   

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        dones[len(dones)-1] = True
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)    

    def PlotModel(self, episode, episodic_reward):
        # If done, the episode has terminated -> save the episode's reward
        self.scores_.append(episodic_reward/self.timestep*self.Training_batch)
        self.episodes_.append(episode)
        # Mean episodic reward of last 50 episodes
        self.average_.append(sum(self.scores_[-50:])/len(self.scores_[-50:]))
        print("Episode * {} * Cur Reward is ==> {}".format(episode,episodic_reward/self.timestep*self.Training_batch))
        print("Episode * {} * Avg Reward is ==> {}".format(episode, sum(self.scores_[-50:])/len(self.scores_[-50:])))        
        
        # if str(episode)[-2:] == "00": 
        if episode % 20 == 0.0:
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.Actor_name+".png")
                print("-----Plots saved-----")
            except OSError:
                pass
        
        # Save the weights every 20 episodes to a file
        if episode % 20 == 0.0:
            self.Actor.Actor.save_weights(self.Actor_name)
            self.Critic.Critic.save_weights(self.Critic_name)

            print("-----Weights saved-----") 

    def go_to_start(self):
        position_reset = PositionTarget()
        position_reset.type_mask = 2496
        position_reset.coordinate_frame = 1
        position_reset.position.x = 0.0
        position_reset.position.y = 0.0
        position_reset.position.z = 2.0
        position_reset.yaw = 0.0
        self.pub_pos.publish(position_reset)          

    def act(self,state):
        # print("State: ", state)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)    
        # print("TF State: ", tf_state)
        pred = self.Actor.predict(tf_state)
        # print("Predict: ", pred)
        # Add exploration (noise)
        action = pred + np.random.uniform(-self.max_action,self.max_action, size = pred.shape)*self.std
        # print("Action: ", action)
        # print("Noise: ",  np.random.uniform(-self.max_action,self.max_action, size = pred.shape)*self.std)
        action = np.clip(action,-self.max_action,self.max_action) # Not exceed bounds
        # print("Clipped Action: ", action)
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)
        # print("Standard Deviation: ",self.std)
        # print("Gaussian Liklihood: ", logp_t)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        pre_sum = -0.5*(((action - pred)/(np.exp(log_std) + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return np.sum(pre_sum, axis=1)    


    def compute_reward(self, state):
        # Compute Reward: use minus because we want to maximize reward
        position_weight = 1.0
        velocity_weight = 0.2 
        angle_weight = 0.05
        reward = -position_weight*(state[0]**2 + state[1]**2)**0.5 # Error in position
        reward += -velocity_weight*(state[2]**2 + state[3]**2)**0.5 # Error in velocity
        reward += -angle_weight*(state[4]**2 + state[5]**2)**0.5 # Error in angle 
        return reward

    def step(self, action):
        #Roll, Pitch in Degrees
        roll_des = action[0,0] 
        pitch_des = action[0,1]
        # print("Action: ", action)
        # print("Roll: ",roll_des)
        # print("Pitch: ",pitch_des)

        #Convert to mavros message and publish desired attitude
        action_mavros = AttitudeTarget()
        action_mavros.type_mask = 7
        action_mavros.thrust = 0.5 # Altitude hold
        action_mavros.orientation = self.rpy2quat(roll_des,pitch_des,0.0)
        self.pub_action.publish(action_mavros)

if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    
    # With eager execution, operations are executed as they are 
    # defined and Tensor objects hold concrete values, which 
    # can be accessed as numpy.ndarray`s through the numpy() method.
    tf.compat.v1.enable_eager_execution()

    PPOAgent()

    r = rospy.Rate(10)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin()        