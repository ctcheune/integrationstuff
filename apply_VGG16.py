#!/usr/bin/env python
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os

import roslib
roslib.load_manifest('faceReco')
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist

navigateTo = None
body = None

def change_target(data):
    global navigateTo
    navigateTo = data.data
    global body 
    body = False

def follow_body(data):
    #global navigateTo
    #navigateTo = data.data
    global body 
    body = True

def callback(data):
    try:
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global body

    if not body:
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            global sess
            global graph
            with graph.as_default():
                tf.keras.backend.set_session(sess)
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                arr = cv2.resize(img[y:y+h,x:x+w],(224,224))
                arr = image.img_to_array(arr)
                arr = np.expand_dims(arr,axis=0)
                arr = utils.preprocess_input(arr,version=1)
                prediction = model.predict(arr)
                prediction = prediction.reshape(1,1,2048)
                finalPred = modelLin.predict(prediction)
                if (names[np.argmax(finalPred)] == navigateTo):
                    if((2*x+w)/2 > 2*(img.shape[0]/3)):
                        vel_msg.angular.z = -0.5
                        vel_msg.linear.x = 0.2 #0
                    elif((2*x+w)/2 < (img.shape[0]/3)):
                        vel_msg.angular.z = 0.5
                        vel_msg.linear.x = 0.2 #0
                    else:
                        vel_msg.angular.z = 0
                        vel_msg.linear.x = 0.2
                    vel_pub.publish(vel_msg)
            cv2.putText(img, names[np.argmax(finalPred)], (x+5,y-5), font, 1, (255,255,255), 2)

    else:
        bodies = body_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in bodies:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            if((2*x+w)/2 > 2*(img.shape[0]/3)):
                vel_msg.angular.z = -0.5
                vel_msg.linear.x = 0.2 #0
            elif((2*x+w)/2 < (img.shape[0]/3)):
                vel_msg.angular.z = 0.5
                vel_msg.linear.x = 0.2 #0
            else:
                vel_msg.angular.z = 0
                vel_msg.linear.x = 0.2
		vel_pub.publish(vel_msg)
		vel_pub.publish(vel_msg)
		vel_pub.publish(vel_msg)

            vel_pub.publish(vel_msg)






sess = tf.Session()
graph = tf.get_default_graph()

tf.keras.backend.set_session(sess)
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
modelLin = tf.keras.models.load_model("/home/osman/catkin_ws/src/FaceReco/faceReco/scripts/pouet.model")

name_file = open("/home/osman/catkin_ws/src/FaceReco/faceReco/scripts/names","rb")
names = pickle.load(name_file)
name_file.close()
print(names)

bridge = CvBridge()
face_detector = cv2.CascadeClassifier('/home/osman/catkin_ws/src/FaceReco/faceReco/scripts/haarcascade_frontalface_default.xml')
body_detector = cv2.CascadeClassifier('/home/osman/catkin_ws/src/FaceReco/faceReco/scripts/haarcascade_lowerbody.xml')
image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle",Image,callback)
name_sub = rospy.Subscriber("/orders/find",String,change_target)
follow_sub = rospy.Subscriber("/orders/follow",String,follow_body)

vel_pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=5)#10
vel_msg = Twist()

#navigateTo = "Osman"
#body = False

vel_msg.linear.x = 0
vel_msg.linear.y = 0
vel_msg.linear.z = 0
vel_msg.angular.x = 0
vel_msg.angular.y = 0
vel_msg.angular.z = 0

font = cv2.FONT_HERSHEY_SIMPLEX

if os.path.isfile("features_file.pickle"):
    prev_feat = open("features_file.pickle","rb")
    feats = pickle.load(prev_feat)
    prev_feat.close()

rospy.init_node('image_converter', anonymous=True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()
