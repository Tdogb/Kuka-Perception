import cv2
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
init.coordinate_units = sl.UNIT.UNIT_METER
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
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        # Display the left image from the numpy array
        cv2.imshow("Image", image_ocv)

def main():
    grabFrames()
    print("Hello World")

if __name__ == "__main__":
    main()