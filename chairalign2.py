#!/usr/bin/env python
import numpy as np
import math
import os
import roslib
roslib.load_manifest('faceReco')
import sys
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

enoughchairs = False

def callback(data):
    global enoughchairs
    people = 2 #hardcoded for now, will fix later
    chairs = data.data
    if len(chairs) >= people:
	    enoughchairs = True
	
    if not enoughchairs:
        vel_msg.angular.z = 0.3
        vel_msg.linear.x = 0.0
        vel_pub.publish(vel_msg)
    else:
        for index in range(len(chairs), -1):

            if chairs[-(index+1)] > 320:
            	vel_msg.angular.z = -0.35
            	vel_msg.linear.x = 0.0 #0

            elif chairs[-(index+1)] < 320:
            	vel_msg.angular.z = 0.35
            	vel_msg.linear.x = 0.0 #0
            else:
                vel_msg.angular.z = 0
            	vel_msg.linear.x = 0
            vel_pub.publish(vel_msg)



    	reason = 'Found chair'
        rospy.signal_shutdown(reason)

    




point_sub = rospy.Subscriber("/chair/point",Int16MultiArray,callback)
speak_sub = rospy.Subscriber("/chair/point",Int16MultiArray,callback)
vel_pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=10)
vel_msg = Twist()
rospy.init_node('chair_find', anonymous=True, disable_signals = True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")


