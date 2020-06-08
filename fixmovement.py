#!/usr/bin/env python

from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import os

import roslib
roslib.load_manifest('faceReco')
import sys
import rospy

from geometry_msgs.msg import Twist


def callback(data):
    vel_msg.linear.x = data.linear.x
    vel_msg.angular.z = data.angular.z
    if vel_msg.angular.z > 0:
	vel_pub.publish(vel_msg)

    else:
    	vel_pub.publish(vel_msg)
    	rospy.sleep(0.15)
    	vel_pub.publish(vel_msg)
    	rospy.sleep(0.15)
    	vel_pub.publish(vel_msg)
    	rospy.sleep(0.15)
    	vel_pub.publish(vel_msg)
    	rospy.sleep(0.15)
    	vel_pub.publish(vel_msg)
	
  

follow_sub = rospy.Subscriber("/vgg16/velocity",Twist,callback)

vel_pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=5)
vel_msg = Twist()



rospy.init_node('move_fixer', anonymous=True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
   
