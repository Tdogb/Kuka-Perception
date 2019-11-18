import time
import math
import sys
import cv2
import numpy as np
import pyzed.sl as sl
import matplotlib.pyplot as plt

usingZed = True
# 3840
img_width = 1920 if usingZed else 1920
img_height = 1080

zed = sl.Camera()

cv2.namedWindow("d", cv2.WINDOW_NORMAL)

tempImage = np.zeros((img_height,img_width,3))
# cv2.imshow("d", tempImage)

def image_callback(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(grayscale,(5,5),0)
    ret, thresh = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((15,15),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Separate into subimages
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    highestContourArea = cv2.contourArea(contours[0])
    subImages = []
    threshedImage = cv2.bitwise_and(image,image,mask=mask)
    #print(contours)
    cv2.drawContours(image, contours, -1, (0,0,255))
    # im2.set_data(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > highestContourArea - (highestContourArea/2):
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            sImg = SubImage(x,y,cv2.getRectSubPix(threshedImage,(w+20,h+20),(x+(w/2),y+(h/2))))
            subImages.append(sImg)
        else:
            break
    #print(subImages[0])
    #im2.set_data(subImages[0].img)
    #print(len(subImages))
    for m in subImages:
        #hsv_thresh = cv2.bitwise_and(hsv,hsv,mask=m)
        #rgb_thresh = cv2.bitwise_and(image,image,mask=m)
        hsv_thresh = cv2.cvtColor(m.img, cv2.COLOR_BGR2HSV)
        threshGreen = cv2.inRange(hsv_thresh,(15,100,10),(35,255,255))
        threshRed = cv2.inRange(m.img, (210,10,10), (255,180,180))
        #divImg = np.divide(m.img[:,:,0],m.img[:,:,1])
        #Blue contour
        xRed,yRed = checkPoint(threshRed)
        #Red contour
        xBlue,yBlue = checkPoint(threshGreen)
        xRed += m.x
        xBlue += m.x
        yRed += m.y
        yBlue += m.y
        cv2.line(image, (xBlue,yBlue), (xRed, yRed), (255,0,0), thickness=3)
        # im2.set_datqa(m.img)
    cv2.imshow("d", image)
        #im2.set_data(hsv)
        # im2.set_data(cv2.cvtColor(hsv_thresh, cv2.COLOR_HSV2RGB))
    # cv2.imwrite("MultipleLetters2.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # time.sleep(1)
class SubImage:
    x = 0
    y = 0
    img = np.array([[0]])
    def __init__(self, _x, _y, _img):
        self.x = _x
        self.y = _y
        self.img = _img


def checkPoint(img_in):
    contours, heirarchy = cv2.findContours(img_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    if len(contours) > 0:
        rect = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x,y,w,h = rect
    return x,y

def zedMain():
    image_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_8U_C4)    
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    while zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        pass
    zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, 10, False)
    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        image_callback(cv2.cvtColor(image_zed.get_data(), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        #plt.pause(0.001)

def staticImageMain():
    # img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/TImage.png")
    img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/MultipleLetters2_edit.png")
    image_callback(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
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