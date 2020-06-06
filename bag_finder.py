#!/usr/bin/env python
import numpy as np
import cv2
import os
import roslib
roslib.load_manifest('faceReco')
import sys 
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Point

global corners

def callback1(data):
    corners = data
   

def callback(data):
    try:
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)

    armrange = 200
    tlCorner = [corners[1], corners[2]]
    brCorner = [corners[3], corners[4]]
    imgvalues = []


    if startmoving = False:
        for row in range(tlCorner[1], brCorner[1], 2):
	    for column in range(tlCorner[2], brCorner[2], 2):
	        value = img[row, column] + ((img[row+1, column]) << 8)
	        imgvalues.append(value)
	        startmoving = True

    if mean(imgvalues) > armrange && startmoving == True:
	vel_msg.linear.x = 0.15
	vel_pub.publish(vel_msg)
    else:
	vel_msg.linear.x = 0
	vel_pub.publish(vel_msg)
	startmoving = False
	usearm = True
    elif usearm = True:
	arm_msg.position.x = mean(imgvalues)
	arm_msg.position.y = 0
	arm_msg.position.z = 0
	arm_pub.publish(arm_msg)
	reason = "arm found handle"
	rospy.signal_shutdown(reason)
    
    startmoving = False
    usearm = False

bridge = CvBridge()

image_sub = rospy.Subscriber("/camera/depth_registered/image_raw",Image,callback)
coord_sub = rospy.Subscriber("/handle/corners", ,callback1)


vel_pub = rospy.Publisher('/handle/move',Twist,queue_size=10)
arm_pub = rospy.Publisher('/handle/arm_move',Point, queue_size=10)
vel_msg = Twist()
arm_msg = Point()




rospy.init_node('bag_finder', anonymous=True, disable_signals=True)



try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
