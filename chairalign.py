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


def callback(data):
    depth = 2400
    angspeed = 0.1
    point = data[0]
    print(point)
    ang = 30
    ang_rad = math.radians(ang)
    width_pix = 240
    width_mm = math.tan(ang_rad) * depth
    fraction = (width - point)/(width)
    point_mm = fraction * width_mm
    angturn = math.atan(point_mm/depth)
    turntime = angturn/angspeed
    timenow = rospy.get_time()
    timelater = timenow + rospy.Duration(turntime)


    rate = rospy.Rate(20)
    while not rospy.is_shutdown() && rospy.get_time() < timelater:
	if fraction > 0:
	    vel_msg.angular.z = angspeed
	    vel_pub.publish(vel_msg)
	elif fraction < 0:
	    vel_msg.angular.z = -angspeed
            vel_pub.publish(vel_msg)
	

    reason = "Found chair"
    rospy.signal_shutdown(reason)

    




point_sub = rospy.Subscriber("/chair/point",Int16MultiArray,callback)
vel_pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=10)
vel_msg = Twist()







rospy.init_node('chair_find', anonymous=True, disable_signals = True)
#try:
    #rospy.spin()
#except KeyboardInterrupt:
    #print("Shutting down")
   
