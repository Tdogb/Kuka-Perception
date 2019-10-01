import cv2
import numpy as np
import pyzed.sl as sl
import rospy
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_QUALITY
init.coordinate_units = sl.UNIT.UNIT_MILLIMETER
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS :
    print(repr(err))
    zed.close()
    exit(1)
image_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
image_ocv = image_zed.get_data()

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

image_pub = rospy.Publisher('image_publisher_zed', Image, queue_size=10)
bridge = CvBridge()

def grabFrames():
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        image_ocv = image_zed.get_data() #numpy array
        ros_img = bridge.cv2_to_imgmsg(image_ocv, "8UC4")
        image_pub.publish(ros_img)

def initProcessing(plane, transform):
    print("Init spatial mapping and camera pose")
    spatial_parameters = sl.SpatialMappingParameters()
    zed.enable_spatial_mapping(spatial_parameters)
    tracking_parameters = sl.TrackingParameters(transform)
    zed.enable_tracking(tracking_parameters)
    #zed.find_floor_plane(plane, transform)

def main():
    rospy.init_node("rosnode_zed")
    signalRecieved = True
    while not signalRecieved:
        pass
    plane = sl.Plane()
    data = np.array([0,0,0,0])
    trasnform_matrix = sl.Matrix4f(data)
    transform = sl.Transform(trasnform_matrix)
    initProcessing(plane, transform)

    while True:
        grabFrames()


if __name__ == "__main__":
    main()