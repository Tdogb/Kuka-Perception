import time
import math
import sys
import cv2
import numpy as np
import pyzed.sl as sl
import matplotlib.pyplot as plt

usingZed = True
# 3840
img_width = 1920 if usingZed else 3840
img_height = 1080

zed = sl.Camera()
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

tempImage = np.zeros((img_height,img_width,3))
im1 = ax1.imshow(tempImage,cmap = 'gray',vmin=0,vmax=255)
im2 = ax2.imshow(tempImage,cmap = 'gray',vmin=0,vmax=255)

plt.ion()

def image_callback(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    blur = cv2.GaussianBlur(grayscale,(5,5),0)
    ret, thresh = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    hsv_thresh = cv2.bitwise_and(hsv,hsv,mask=mask)
    thresh2 = cv2.inRange(hsv_thresh,(40,50,175),(80,255,255))
    im1.set_data(mask)
    im2.set_data(thresh2)

def zedMain():
    image_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_8U_C4)    
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    while zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        pass
    zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, 30, False)
    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        image_callback(image_zed.get_data())
        plt.pause(0.001)

def staticImageMain():
    img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/TImage.png")
    image_callback(img)
    #im2.set_data(img)

def initCamera():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
    init.coordinate_units = sl.UNIT.UNIT_METER
    if len(sys.argv) >= 2 :
        init.svo_input_filename = sys.argv[1]

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
    zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, -1, True)

if __name__ == "__main__":
    if usingZed:
        zed.close()
        initCamera()
        zedMain()
    else:
        staticImageMain()
    plt.ioff()
    plt.show()