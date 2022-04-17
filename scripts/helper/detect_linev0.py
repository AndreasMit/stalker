#!/usr/bin/env python3 

import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #convert Image to CV format
from color_detector.msg import PREDdata

class line_detector:

  def __init__(self):
    self.image_pub = rospy.Publisher("/detection",Image,queue_size=10)
    self.box_pub = rospy.Publisher("/box", PREDdata , queue_size=10)
    self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw",Image,self.callback) 
    self.bridge = CvBridge()
    self.box = PREDdata()
    # self.angle_prev = 0

  def callback(self,data):
    try:
      image = self.bridge.imgmsg_to_cv2(data,'bgr8')
    except CvBridgeError as e:
      print(e)

    # (rows,cols,channels) = image.shape
    # center = (int(cols/2-1),int(rows/2-1)) # center of the image

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #8 bit have 255 as max value, H has 180 max, we want high V that is gray (as the color of the pavement)
    gray_mask = cv.inRange(image_hsv, (0,0,25), (0,0,255) ) 

    #check if that mask matches exactly the pavement
    # gray_mask_r = cv.bitwise_not(gray_mask) #invert zeros to ones 
    # im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # to gray
    # im_left = gray_mask_r*im_gray # mask gray image
    # cv.imshow("masked",im_left)
    cv.waitKey(3)

    contours, _ = cv.findContours(gray_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0: 
      areas = [cv.contourArea(c) for c in contours]
      index = np.argmax(areas)
      area = areas[index]
      if area <200000: #we want appropriate altitude to detect path thats why we check area
        #maybe also check area>200
        box = cv.minAreaRect(contours[index])
        # center_box, (width, height), angle = box
        box = cv.boxPoints(box)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0, 0, 255), 1)
        # cv.line(image, (int(center_box[0]),int(center_box[1])), center, (255, 0, 0), 2)
    
        # cv.imshow("box", image)

        
        self.box.box_1 = box[0][:]
        self.box.box_2 = box[1][:]
        self.box.box_3 = box[2][:]
        self.box.box_4 = box[3][:]
        
        # cv.waitKey(3)
        # cv.imwrite('ol.jpg', image)
        # print(image)
        # distance = np.linalg.norm(np.array(center)-np.array(center_box))
        # angle = abs(angle)
        # angle_new = 0
        # if self.angle_prev > 80 and angle < 10:
        #   angle_new = angle + 90
        #   self.angle_prev = angle
        # elif self.angle_prev < 10 and angle > 80:
        #   angle_new = angle - 90
        #   self.angle_prev = angle
        # else:
        #   angle_new = angle
        # print(angle)

    else:
      print('out of bounds')
      self.box.box_1 = 0
      self.box.box_2 = 0
      self.box.box_3 = 0
      self.box.box_4 = 0

    #publish the predicted data, send 0 if no valid detection
    self.box_pub.publish(self.box)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
    except CvBridgeError as e:
      print(e)


if __name__ == '__main__':
  ic = line_detector()
  rospy.init_node('box_detector', anonymous=True)
  print('starting node')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()
