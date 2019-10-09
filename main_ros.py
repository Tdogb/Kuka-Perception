import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import math

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

upper_hsv_bounds = np.array([188,40,65])
lower_hsv_bounds = np.array([153,0,0])

camera_angle = -math.pi/4

depth_mask = np.zeros((720,1280,3), np.uint8)

def image_callback(data):
    alpha_image = bridge.imgmsg_to_cv2(data)
    image_tmp = alpha_image[:,:,:3]
    image = cv2.bitwise_and(depth_mask.astype(np.uint8), image_tmp)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_threshold_rgb = cv2.inRange(image, lower_color_bounds, upper_color_bounds)
    image_threshold_hsv = cv2.inRange(image, lower_hsv_bounds, upper_hsv_bounds)
    image_threshold = cv2.bitwise_and(cv2.bitwise_not(image_threshold_hsv), image_threshold_rgb)

    #image_threshold_hsv_2 = cv2.threshold(hsv_image, 10, 255, cv2.THRESH_TOZERO)
    ros_image = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)

def depth_callback(data):
    depth_map = bridge.imgmsg_to_cv2(data)
    depth_map_grayscale = depth_map * 100
    #print(depth_map_grayscale)
    #depth_map_grayscale = (depth_map*50).astype(np.uint8)
    #print(depth_map)

    ret, depth_mask = cv2.threshold(depth_map_grayscale, 225, 255, cv2.THRESH_BINARY)
    # for x in range(0,720):
    #     for y in range(0,1280):
    #         #print(depth_map[x][y])
    #         if(depth_map[x][y] < 2.1):
    #             depth_mask[x][y] = 255
    #         else:
    #             depth_mask[x][y] = 0
    #ros_image = bridge.cv2_to_imgmsg(depth_mask)

    #image_pub.publish(ros_image)

rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, image_callback)
depth_sub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_callback)
#pointcloud_sub = rospy.Subscriber("/zed/zed_node/point_cloud/fused_cloud_registered", PointCloud2, pointcloud_callback)

if __name__ == "__main__":
    rospy.spin()

    #contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
