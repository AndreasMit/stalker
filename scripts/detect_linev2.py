#!/usr/bin/env python3 

import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #convert Image to CV format
from color_detector.msg import PREDdata

class box_detector:

  def __init__(self):
    self.image_pub = rospy.Publisher("/Detection",Image,queue_size=10)
    self.box_pub = rospy.Publisher("/box", PREDdata , queue_size=10)
    self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw",Image,self.callback) 
    self.bridge = CvBridge()
    self.box = PREDdata()

  def callback(self,data):
    try:
      image = self.bridge.imgmsg_to_cv2(data,'bgr8')
    except CvBridgeError as e:
      print(e)

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #8 bit have 255 as max value, H has 180 max, we want high V that is gray (as the color of the pavement)
    gray_mask = cv.inRange(image_hsv, (0,0,25), (0,0,255) ) 

    contours, _ = cv.findContours(gray_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    area = 0
    if len(contours) > 0: 
      areas = [cv.contourArea(c) for c in contours]
      index = np.argmax(areas)
      area = areas[index]

    if len(contours) > 0 and area>200 and area<200000 : #we want appropriate altitude to detect path thats why we check area
      box = cv.minAreaRect(contours[index])
      box = cv.boxPoints(box)
      box = np.int0(box)
      cv.drawContours(image, [box], 0, (0, 0, 255), 1)
      # cv.imshow("box", image)
      # cv.waitKey(30)
      
      self.box.box_1 = box[0][:]
      self.box.box_2 = box[1][:]
      self.box.box_3 = box[2][:]
      self.box.box_4 = box[3][:]

    else:
      # print('out of bounds')
      self.box.box_1 = [0,0]
      self.box.box_2 = [0,0]
      self.box.box_3 = [0,0]
      self.box.box_4 = [0,0]

    #publish the predicted data, send 0 if no valid detection
    self.box_pub.publish(self.box)
    # print(self.box)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
    except CvBridgeError as e:
      print(e)


if __name__ == '__main__':
  ic = box_detector()
  rospy.init_node('box_detector', anonymous=True)
  print('starting node')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv.destroyAllWindows()
