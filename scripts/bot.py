#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import numpy as np
import time
import random
import math
import signal

# maybe use this topic to create trajectories?
# /move_base_simple/goal - geometry_msgs/PoseStamped


def publishVelocities(u, omega):
    velMsg = Twist()

    velMsg.linear.x = u
    velMsg.linear.y = 0.0
    velMsg.linear.z = 0.0

    velMsg.angular.x = 0.0
    velMsg.angular.y = 0.0
    velMsg.angular.z = omega

    # Publish velocities #
    pub_vel.publish(velMsg)

# Stop robot #
def stopLeader():
    velMsg = Twist()

    velMsg.linear.x = 0.0
    velMsg.linear.y = 0.0
    velMsg.linear.z = 0.0

    velMsg.angular.x = 0.0
    velMsg.angular.y = 0.0
    velMsg.angular.z = 0.0

    count = 0
    while count < 30:
        pub_vel.publish(velMsg)
        count += 1


def sigHandler(num, frame):

    print("Signal occurred:  " + str(num))

    # Stop robot #
    stopLeader()

    # Close ros #
    rospy.signal_shutdown(0)
    exit()



if __name__=='__main__':
    rospy.init_node('bot', anonymous=True)

    pub_vel = rospy.Publisher('/robot/robotnik_base_control/cmd_vel/',Twist,queue_size=10000)

    # Set sig handler for proper termination #
    signal.signal(signal.SIGINT, sigHandler)
    signal.signal(signal.SIGTERM, sigHandler)
    signal.signal(signal.SIGTSTP, sigHandler)

    linearVel = 0.55
    endTime = time.time() + 60 * 0.1 # Run for 4 min the experiment

    # Create bucket of angular velocities #
    samplesAngular = set()
    num = 0.4
    while num <= 2.0:
        samplesAngular.add(num)
        num += 0.5

    # Pick initial random angular velocity #
    angularVel = random.sample(samplesAngular, 1)[0]

    count = 0
    # Take random actions #
    while time.time() < endTime:

        # Expondential decay #
        if abs(angularVel) > 0.05:
            angularVel *= 0.999977
        else:
            # Reset angular velocity #
            angularVel = random.sample(samplesAngular, 1)[0]

            # Change sign #
            count += 1
            if count == 1:
                angularVel *= -1
                count = 0

        print("Bot(Leader): (uL: {} m/s, omegaL: {} r/s)".format(linearVel, angularVel))
        # Push commands #
        publishVelocities(linearVel, angularVel)

    # Stop movement #
    stopLeader()
