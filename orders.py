#!/usr/bin/env python
import roslib
roslib.load_manifest('faceReco')
import sys
import rospy
from std_msgs.msg import String
import pickle


def callback(data):
    if "FIND" in data.data:
        print(data.data)
        count = 0
        while(count < len(names) and not (names[count].upper() in data.data)):
            count += 1
        if(count < len(names)):
            orders_find.publish(names[count])

    if "FOLLOW" in data.data:
        print(data.data)
        orders_follow.publish("follow body")

    if "STOP" in data.data:
        print(data.data)
        stop_follow.publish("stop")


name_file = open("/home/osman/catkin_ws/src/FaceReco/faceReco/scripts/names","rb")
names = pickle.load(name_file)
name_file.close()

orders_find = rospy.Publisher('/orders/find',String,queue_size=10)
orders_follow = rospy.Publisher('/orders/follow',String,queue_size=10)
orders_follow = rospy.Publisher('/orders/stop',String,queue_size=10)
talk_sub = rospy.Subscriber('/grammar_data',String,callback)

rospy.init_node('orders_node', anonymous=True)

try:
    rospy.spin()
except rospy.ROSInterruptException:
    print("Shutting down")
