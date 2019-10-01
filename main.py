import cv2
import numpy as np
import pyzed.sl as sl

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

def grabFrames():
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        image_ocv = image_zed.get_data() #numpy array
        cv2.imshow("Image", image_ocv)

def initProcessing(plane, transform):
    print("Init spatial mapping and camera pose")
    zed.enable_spatial_mapping()
    zed.enable_tracking()
    zed.find_floor_plane(plane, transform)

def main():
    signalRecieved = True
    while not signalRecieved:
    
    plane = sl.Plane()
    transform = sl.Transform()
    initProcessing(plane, transform)

    while True:
        grabFrames()


if __name__ == "__main__":
    main()