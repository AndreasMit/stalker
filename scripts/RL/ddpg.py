#!/usr/bin/env python3

import rospy
import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.keras import layers

from UOAnoise import OUActionNoise
from models import Models
from environment import Environment
from buffer import Buffer

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau)) 


if __name__=='__main__':
    rospy.init_node('rl_node', anonymous=True)
    
    # With eager execution, operations are executed as they are 
    # defined and Tensor objects hold concrete values, which 
    # can be accessed as numpy.ndarray`s through the numpy() method.
    tf.compat.v1.enable_eager_execution()


    ########## variables ##############
    num_actions = 3 #roll,pitch and yaw
    num_states = 3  

    angle_max = 2.0 # constraints for commanded roll and pitch
    yaw_max = 5 #how much yaw should change every time
    max_vel_up = 1.5 # Real one is 2.5
    angle_min = -angle_max 
    yaw_min = -yaw_max
    max_vel_down = -max_vel_up # constraints for commanded vertical velocity

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005 

    # buffer = Buffer(100000, 1000)
    buffer = Buffer(100000, 64)

    std_dev = 0.1
    # std_dev = 0.2

    models = Models(num_states, num_actions, angle_max, yaw_max)
    actor_model = models.get_actor()
    print("Actor Model Summary")
    print(actor_model.summary())

    critic_model = models.get_critic()
    print("Critic Model Summary")
    print(critic_model.summary())

    target_actor = models.get_actor()
    target_critic = models.get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Load pretrained weights
    # actor_model.load_weights('ddpg_actor.h5')
    # critic_model.load_weights('ddpg_critic.h5')

    # target_actor.load_weights('ddpg_target_actor.h5')
    # target_critic.load_weights('ddpg_target_critic.h5')

    # Define optimizer
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = [] 
    episodes = []
   
    Environment()

    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

    r = rospy.Rate(20)
    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin() 