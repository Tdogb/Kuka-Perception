import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

upper_hsv_bounds = np.array([188,40,65])
lower_hsv_bounds = np.array([153,0,0])
def image_callback(data):
    alpha_image = bridge.imgmsg_to_cv2(data)
    image = alpha_image[:,:,:3]
    #Table location
    mask = np.zeros(image.shape, dtype=np.uint8)
    contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #end of table mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_threshold_rgb = cv2.inRange(image, lower_color_bounds, upper_color_bounds)
    image_threshold_hsv = cv2.inRange(image, lower_hsv_bounds, upper_hsv_bounds)
    image_threshold = cv2.bitwise_and(cv2.bitwise_not(image_threshold_hsv), image_threshold_rgb)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    ros_image = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)

rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, image_callback)

def main():
    print("")

if __name__ == "__main__":
    rospy.spin()
    #while True:
     #   main()
      #  time.sleep(0.05)