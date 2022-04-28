self.current_state = np.array([self.distance , self.angle , self.desired_pos_z-self.z_position, self.desired_vel_x-self.x_velocity,
                                            # self.x_velocity , self.y_velocity, self.z_velocity, 
                                            self.roll, self.pitch, self.yaw, 
                                            # np.rad2deg(self.x_angular), np.rad2deg(self.y_angular), 
                                            self.previous_action[0], self.previous_action[1], self.previous_action[2], self.previous_action[3]])


if self.timestep > 1:

                #REWARD
                max_distance = 10000 #pixels
                max_distance_up = 0.5 #meters
                position_error = abs(self.current_state[0])/max_distance+abs(self.current_state[1])/90 +abs(self.current_state[2])/max_distance_up
                weight_position = 1.6

                # Oscillation suppression -> smooth output action
                delta_roll = abs(self.action[0]-self.current_state[11])/angle_max # normalized -> max movement from previous to current action is 2 (e.g from -10 to 10)
                delta_pitch = abs(self.action[1]-self.current_state[12])/angle_max
                delta_yaw = abs(self.action[2]-self.current_state[13])/angle_max
                delta_zdot = abs(self.action[3]-self.current_state[14])/max_vel_up
                delta_action = delta_roll + delta_pitch + delta_yaw + delta_zdot     
                weight_smoothness = 0.30

                delta_roll = abs(self.action[0]-self.previous_action[0])/angle_max # normalized -> max movement from previous to current action is 2 (e.g from -10 to 10)
                delta_pitch = abs(self.action[1]-self.previous_action[1])/angle_max
                delta_yaw = abs(self.action[2]-(self.previous_action[2]))/yaw_max #[-90,90] action but i publish it with +90 -> [-180,180]
                delta_action = delta_roll + delta_pitch + delta_yaw     
                weight_smoothness = 0.30
        
                action = abs(self.action[0])/angle_max + abs(self.action[1])/angle_max + abs(self.action[2])/angle_max + abs(self.action[3])/max_vel_up
                weight_action = 0.10/max(position_error,0.01)

                velocity_error = abs(current_state[3])/max_vel_up
                weight_velocity = 0.10

                #use minus because we want to maximize reward
                self.reward = -weight_position*position_error 
                self.reward += - weight_smoothness*delta_action
                self.reward += -weight_action*action
                self.reward += -weight_velocity*velocity_error
               
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