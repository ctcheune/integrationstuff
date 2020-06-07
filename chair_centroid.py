# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import rospy
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image

THRESHOLD = 0.3
CONFIDENCE = 0.5
ChairClass = 56

# load the COCO class labels our YOLO model was trained on
labelsPath = "darknet/data/coco.names" #os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "darknet/yolov3.weights" #os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath =  "darknet/cfg/yolov3.cfg" #os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def chair_callback(image):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    chair_centroids = []
    chair_data = Int16MultiArray()


    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            if classID != ChairClass:
                continue
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
    if confidence > CONFIDENCE:
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        # update our list of bounding box coordinates, confidences,
        # and class IDs
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)


    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
        THRESHOLD)


    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            chair_centroids.append(int(x+w/2.0))

    #Check there is data to publish, sort and publish
    if chair_centroids:
        chair_centroids.sort()
        chair_data.data = chair_centroids
        chair_pub.publish(chair_data)

chair_pub = rospy.Publisher("/chair/point",Int16MultiArray,queue_size=10)
chair_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle2", Image, chair_callback)

rospy.init_node('image_converter', anonymous=True)
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()