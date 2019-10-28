import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import math
import tensorflow as tf

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)
image_pub2 = rospy.Publisher("image_2", Image, queue_size=10)

img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0

def image_callback(data):
    global filenum
    alpha_image = bridge.imgmsg_to_cv2(data)
    image_large = alpha_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # image_threshold_hsv = cv2.inRange(image, (16,61,98), (33,255,255))
    image_threshold_hsv = cv2.inRange(image, (0,61,98), (33,255,255))

    contour_image = np.uint8(image_threshold_hsv)
    contours, heirarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sub_images = [None] * len(contours)
    i = 0
    contouri = 0
    print(contours)
    for c in contours:
        if cv2.contourArea(contours[contouri]) > 50:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            if w > h:
                h = w
            if h > w:
                w = h
            sub_images[i] = cv2.getRectSubPix(image,(w+20,h+20),(x+(w/2),y+(h/2)))
            sub_images[i] = cv2.resize(sub_images[i], (64,64))
            #cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
            i+=1
        contouri +=1
    # Begin ICP

    

    # End ICP

    #cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    ros_image = bridge.cv2_to_imgmsg(sub_images[0])
    ros_image2 = bridge.cv2_to_imgmsg(image_threshold_hsv)
    image_pub.publish(ros_image)
    image_pub2.publish(ros_image2)
    #print(filenum)
    filename = 'I.' + str(filenum) + '.png'
    filenum+=1  
    #cv2.imwrite(filename,sub_images[0])
    time.sleep(0.25)

rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, image_callback)

if __name__ == "__main__":
    rospy.spin()