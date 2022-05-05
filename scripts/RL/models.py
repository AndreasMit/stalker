#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.keras import layers

class Models:
    def __init__(self, num_states, num_actions, angle_max, yaw_max):
        self.num_states = num_states
        self.num_actions = num_actions
        self.angle_max = angle_max
        self.yaw_max = yaw_max

    def get_actor():

        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        h1 = layers.Dense(128, activation="tanh")(inputs)
        h2 = layers.Dense(128, activation="tanh")(h1)    
        outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(h2)

        # Output of tanh is [-1,1] so multiply with the upper control action
        outputs = outputs * [self.angle_max, self.angle_max, self.yaw_max]
            
        model = tf.keras.Model(inputs, outputs)

        return model  

    def get_critic():

        # The critic NN has 2 inputs: the states and the actions. Use 2 seperate NN and then concatenate them
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        h1_state = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(h1_state)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(128, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model 