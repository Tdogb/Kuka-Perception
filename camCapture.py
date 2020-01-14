import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import math

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)
camera_angle = -math.pi/4
depth_mask = np.zeros((720,1280,3), np.uint8)
img_width = 384
img_height = 216
points_image = np.array([[0,0],[0,0],[0,0]])
points_image_width = 0
points_image_height = 0
points_depth = np.array([[0.],[0.],[0.]])
depth_width = 0.0
depth_height = 0.0

def image_callback(data):
    alpha_image = bridge.imgmsg_to_cv2(data)
    image_large = alpha_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    #image = cv2.bitwise_and(depth_mask.astype(np.uint8), image_tmp)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image_threshold_rgb = cv2.inRange(image, lower_color_bounds, upper_color_bounds)
    #image_threshold_hsv = cv2.inRange(image, (18,61,158), (31,255,255))
    #image_threshold = cv2.bitwise_and(cv2.bitwise_not(image_threshold_hsv), image_threshold_rgb)

    #contour_image = np.uint8(image_threshold_hsv)
    #contours, heirarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)depth_vectors = np.array([np.array([0,0]),np.array([0,0])])

    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    #depth_vectors = np.array([np.array([0,0]),np.array([0,0])])

    # matrix_rot = cv2.getRotationMatrix2D((img_width/2, img_height/2), 1, 1)
    # tranformed_image = cv2.warpPerspective(image, matrix_rot, (img_width*2, img_height*2))

    #matrix_rot = cv2.getPerspectiveTransform()
    # if np.size(contours) > 0:
    #     x,y,w,h = cv2.boundingRect(contours[0])
    #     points_image[0] = np.array([x,y])
    #     points_image[1] = np.array([x+w,y])
    #     points_image[2] = np.array([x,y+h])
    #     points_image_height = h
    #     points_image_width = w
    #     print(points_image)

    image_threshold_hsv_2 = cv2.threshold(hsv_image, 10, 255, cv2.THRESH_TOZERO)
    ros_image = bridge.cv2_to_imgmsg(image_large)
    image_pub.publish(ros_image)
    cv2.imwrite('sticker.png', image_large)
    time.sleep(5)

def depth_callback(data):
    depth_map = bridge.imgmsg_to_cv2(data)
    for i in range(0,2):
        points_depth[i] = depth_map[points_image[i,0],points_image[i,1]]
    depth_vectors = np.array([np.array([points_depth[0],points_depth[1]]),np.array([points_depth[0],points_depth[2]])])
    depth_height = np.linalg.norm(depth_vectors[0])
    depth_width = np.linalg.norm(depth_vectors[1])

    depth_height_unit_vector = depth_vectors[0]/depth_height
    depth_width_unit_vector = depth_vectors[1]/depth_width

    desired_image_height = 300
    desired_image_width = desired_image_height * (points_image_width/points_image_height)

    rect_depth = np.array([])


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
#depth_sub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_callback)

if __name__ == "__main__":
    rospy.spin()

    #contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
