import time
import sys
import rospy
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
import pyzed.sl as sl
import tensorflow as tf
from tensorflow import keras

usingZed = True
img_width = 1920 if usingZed else 1920
img_height = 1080
zed = sl.Camera()
#cv2.namedWindow("d", cv2.WINDOW_NORMAL)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

tempImage = np.zeros((img_height,img_width,3))
im1 = ax1.imshow(tempImage,cmap = 'gray',vmin=0,vmax=255)
im2 = ax2.imshow(tempImage,cmap = 'gray',vmin=0,vmax=255)
img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0

alphabetModel = keras.models.load_model('alphabetTrained26.h5')
print("Finished loaded trained weights")


testImgDir = '/home/tannerliu/Desktop/tensorFlow/Alphabet_implementation/test_images'
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
plt.ion()
def prepare(img):
    IMG_HEIGHT = 50
    imgArr = img
    imgArr = cv2.resize(imgArr, (IMG_HEIGHT, IMG_HEIGHT))
    lowerBound = np.array([20, 100, 130])
    upperBound = np.array([250, 255, 255])
    maskImg = cv2.inRange(imgArr, lowerBound, upperBound)
    return maskImg.reshape(-1, IMG_HEIGHT, IMG_HEIGHT, 1)


def image_callback(data):
    global filenum
    alpha_image = data
    image_large = alpha_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # image_threshold_hsv = cv2.inRange(image, (60,120,180), (255,255,255))
    image_threshold_hsv = cv2.inRange(image, (60,120,180), (255,255,255))
    contour_image = np.uint8(image_threshold_hsv)
    contours, heirarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #sub_images = [None] * len(contours)
    sub_images = []
    # mask_images = []
    i = 0
    contouri = 0
    output_contour_index = 0
    for c in contours:
        if heirarchy[0,contouri,3] == -1 and cv2.contourArea(contours[contouri]) > 30:
            output_contour_index = contouri
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            if w > h:
                h = w
            if h > w:
                w = h
            tempMask = np.zeros_like(image)
            cv2.drawContours(tempMask, contours, contouri, (255, 255, 255), thickness=cv2.FILLED)
            tempImage = cv2.bitwise_and(image,image,mask=tempMask[:,:,0])
            img = cv2.getRectSubPix(tempImage,(w+20,h+20),(x+(w/2),y+(h/2)))
            # mask = cv2.getRectSubPix(mask,(w+20,h+20),(x+(w/2),y+(h/2)))
            # sMaskImg = SubImage(x,y,mask)
            sImg = SubImage(x,y,img)
            sub_images.append(sImg)
            # mask_images.append(sMaskImg)
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
            i+=1
        contouri +=1
    # ros_image = bridge.cv2_to_imgmsg(sub_images[0].img)
    for m in sub_images:
        if m is not None:
            # ratioMaskTemp = cv2.inRange(m.img, (15,15,15), (255,255,255))
            #ratioMask = cv2.threshold(ratioMaskTemp, 1, 255, cv2.THRESH_BINARY)
            ratioMask = m.img[:,:,1] // m.img[:,:,0]
            ratioMask2 = m.img[:,:,0] // m.img[:,:,1]
            #print(ratioMask)
            hsv_thresh = cv2.cvtColor(m.img, cv2.COLOR_BGR2HSV)
            ratioTemp = m.img[:,:,0] / m.img[:,:,1]
            ratioTemp2 = m.img[:,:,1] / m.img[:,:,0]
            ratioTemp *= 255
            ratioTemp2 *= 255
            ratio = ratioTemp.astype(np.uint8)

            # ratio = cv2.bitwise_and(ratio, ratio, mask=ratioMask)
            # ratio = cv2.bitwise_not(ratio,ratioMask)

            ratio2 = ratioTemp2.astype(np.uint8)
            # ratio2 = cv2.bitwise_and(ratio2, ratio2, mask=ratioMask2)
            # ratio2 = cv2.bitwise_not(ratio2,cv2.bitwise_not(ratioMask2))

            # threshGreen = cv2.inRange(hsv_thresh,(15,40,10),(70,255,255))
            # threshRed = cv2.inRange(hsv_thresh, (100,175,140),(150,255,255))#, (210,10,10), (255,180,180))
            threshRed = cv2.inRange(m.img, (0,0,220), (50,130,255))
            threshGreen = cv2.inRange(m.img, (0,75,0), (110,255,80))            
            
            #divImg = np.divide(m.img[:,:,0],m.img[:,:,1])
            #Blue contour
            xRed,yRed = checkPoint(threshRed)
            #Red contour
            xBlue,yBlue = checkPoint(threshGreen)
            xRed += m.x
            xBlue += m.x
            yRed += m.y
            yBlue += m.y
            cv2.line(image, (xBlue,yBlue), (xRed, yRed), (0,255,0), thickness=2)
            prepedImg = np.float32(prepare(cv2.resize(m.img,(64,64))))
            #prepedImg = np.float32(prepare(os.path.join(testImgDir, 'I/I.194.png')))
            prediction = alphabetModel.predict([prepedImg])
            #print(CATEGORIES[int(prediction[0][0])])
            #print(np.argmax(prediction[0]))
            #print(CATEGORIES[np.argmax(prediction[0])])
    # filename = 'TestImage' + str(filenum) + '.png'
    filename = 'TestDots2.png'
    filenum+=1
    im1.set_data(image)
    #cv2.imshow("d", image)
    # cv2.imwrite(filename,image)
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
        plt.pause(0.001)

def staticImageMain():
    # img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/TImage.png")
    img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/MultipleLetters2_edit.png")
    image_callback(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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