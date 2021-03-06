#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import numpy as np
import time
import random
import math
import signal
from stalker.msg import PREDdata

# maybe use this topic to create trajectories?
# /move_base_simple/goal - geometry_msgs/PoseStamped

class controller:

    def __init__(self):
        self.pub_vel = rospy.Publisher('/robot/robotnik_base_control/cmd_vel/',Twist,queue_size=1) #queue size 1 so that we can stop the execution when we want
        self.sub_detector = rospy.Subscriber("/box", PREDdata, self.box_callback)
        self.outofbounds = False
        self.prev = False
        self.linearVel = 0.5 #m/s
        # self.angularVel = random.sample([0.4, 0.9, 1.4, 1.9], 1)[0]
        # self.angularVel = random.sample([0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2], 1)[0]
        # self.angularVel = random.sample([0.2, 0.5, 0.7, 1.0], 1)[0]
        self.angularVel = random.sample([0.2, 0, 0.5, 0, 0.7, 0, 1.0], 1)[0]
        # Set sig handler for proper termination #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTERM, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

    def control(self):
        sign = 0
        counter = 0
        while True:

            if self.outofbounds == False:
                if self.prev == True: #if we didnt have a box and now we do have it, wait a bit so that the drone can align over it before starting moving again.
                    time.sleep(5)
                    counter = 0
                    
                else:
                    # Expondential decay #
                    if abs(self.angularVel) > 0.05:
                        self.angularVel *= 0.999977
                    else:
                        # Reset angular velocity 
                        self.angularVel = random.sample([0.2, 0.5, 0.7, 1.0], 1)[0]
                        # self.angularVel = random.sample([0.2, 0, 0.5, 0, 0.7, 0, 1.0], 1)[0]
                        if sign == 0:
                            self.angularVel = self.angularVel
                            sign = 1
                        else:
                            self.angularVel = (-1)*self.angularVel
                            sign = 0
                        
                        # a random stop in between timesteps in case we lose summit
                        if counter == 30 :
                            print('random stop')
                            self.stopLeader()
                            time.sleep(20)

                            counter = 0
                
                        counter += 1
                    # print(self.angularVel)
                    # print("Bot(Leader): (uL: {} m/s, omegaL: {} r/s)".format(self.linearVel, self.angularVel))
                    self.publishVelocities(self.linearVel, self.angularVel)
            else:
                # print('stopping leader')
                self.stopLeader()


    def box_callback(self,box):
        if self.outofbounds == True:
            self.prev = True
        else:
             self.prev = False

        if box.box_1==(0,0) and box.box_2==(0,0) and box.box_3==(0,0) and box.box_4==(0,0):
            self.outofbounds = True
        else:
            self.outofbounds = False
      
    def publishVelocities(self, u, omega):
        velMsg = Twist()
        velMsg.linear.x = u
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0
        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = omega
        # Publish velocities #
        self.pub_vel.publish(velMsg)

    # Stop robot #
    def stopLeader(self):
        velMsg = Twist()
        velMsg.linear.x = 0.0
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0
        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = 0.0

        for _ in range(30):
            self.pub_vel.publish(velMsg)


    def sigHandler(self, num, frame):
        print("Signal occurred:  " + str(num))
        # Stop robot 
        self.stopLeader()
        # Close ros
        # rospy.signal_shutdown(0)
        exit()


if __name__=='__main__':
    rospy.init_node('bot', anonymous=True)
    print('starting bot')
    c = controller()
    c.control()

    while not rospy.is_shutdown:
        r.sleep()    

    rospy.spin() 