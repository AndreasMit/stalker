#!/usr/bin/env python3 

import roslib
roslib.load_manifest('stalker')
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #convert Image to CV format

red = (153, 0, 18)
dim= (720, 480) 

class line_detector:

  def __init__(self):
    self.image_pub = rospy.Publisher("/detection",Image,queue_size=10)
    self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw",Image,self.callback) 
    self.bridge = CvBridge()

  def callback(self,data):
    print('new image')
    try:
      image = self.bridge.imgmsg_to_cv2(data,'bgr8')
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = image.shape
    print(image.shape)

    def create_blank(width, height, rgb_color=(0, 0, 0)):
      image = np.zeros((height, width, 3), np.uint8)
      color = tuple(reversed(rgb_color))
      image[:] = color
      return image

    if cols > 60 and rows > 60 :
      cv.circle(image, (50,50), 10, 255)
      cv.imshow("Image window", image)
      cv.waitKey(3)
      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
      except CvBridgeError as e:
        print(e)

    # copy_image = np.copy(cv_image)
    # copy_mask = np.copy(segimg)
    # ww = copy_mask.shape[1]
    # hh = copy_mask.shape[0]
    # red_mask = create_blank(ww, hh, rgb_color=red)
    # copy_mask=cv2.bitwise_and(copy_mask,red_mask)
    # combo_image=cv2.addWeighted(copy_image, 1, copy_mask,1 ,1)

    # mask = cv2.inRange(segimg, (130, 130, 130), (255, 255, 255))
    # kernel = np.ones((1, 1), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    # contours_blk, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # _, contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # print("contours: ", contours_blk)
    # areas = [cv2.contourArea(c) for c in contours_blk]
    # max_index = np.argmax(areas)
    # #cnt = countours_blk[max_index]
    # #contours_blk.sort(key=cv2.minAreaRect)
    # if len(contours_blk) > 0 and cv2.contourArea(contours_blk[max_index]) > 200:
    #       # Box creation for the detected coastline
    #         blackbox = cv2.minAreaRect(contours_blk[max_index])
    #         (x_min, y_min), (w_min, h_min), angle = blackbox            
    #         box = cv2.boxPoints(blackbox)
    #         box = np.int0(box)
    #         (x_min, y_min), (w_min, h_min), angle = blackbox
    #         box = cv2.boxPoints(blackbox)
    #         box = np.int0(box)
    #         # print("1st point of box: ", box[0][:])
    #         # print("2nd point of box: ", box[1][:])
    #         # print("3rd point of box: ", box[2][:])
    #         # print("4th point of box: ", box[3][:]) 

    #         pred_data = PREDdata()
    #         pred_data.box_1 = box[0][:]
    #         pred_data.box_2 = box[1][:]
    #         pred_data.box_3 = box[2][:]
    #         pred_data.box_4 = box[3][:]
    #         self.pub_pred_data.publish(pred_data)
    #         # cv2.imshow("Combined prediction", combo_image)
    #         # cv2.imshow("Prediction image window", segimg)   
    #         cv2.waitKey(3)

    # try:
    #   open_cv_image = self.bridge.cv2_to_imgmsg(segimg, "bgr8")
    #   open_cv_image.header.stamp = data.header.stamp
    #   self.image_pub_first_image.publish(open_cv_image)
      
    #   combo_open_cv_image = self.bridge.cv2_to_imgmsg(combo_image, "bgr8")
    #   combo_open_cv_image.header.stamp = data.header.stamp
    #   self.image_pub_second_image.publish(combo_open_cv_image)
      
    #   cv2.drawContours(segimg, [box], 0, (0, 0, 255), 1)
    #   cv2.line(segimg, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)
    #   # cv2.drawContours(combo_image, [box], 0, (0, 0, 255), 1)
    #   # cv2.line(combo_image, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)
      
    #   cv_image = cv2.resize(segimg, (720, 480))
    #   ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
    #   ros_image.header.stamp = data.header.stamp
    #   self.ros_image_pub.publish(ros_image)
    # except CvBridgeError as e:
    #   print(e)


if __name__ == '__main__':
  ic = line_detector()
  rospy.init_node('line_detector', anonymous=True)
  print('starting node')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()
