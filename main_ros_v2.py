import time
import rospy
import cv2
import numpy as np
import subprocess
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import math
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from tensorflow import keras

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)
image_pub2 = rospy.Publisher("image_2", Image, queue_size=10)
image_pub3 = rospy.Publisher("image_3", Image, queue_size=10)

img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0

alphabetModel = keras.models.load_model('alphabetTrained26.h5')
print("Finished loaded trained weights")


testImgDir = '/home/tannerliu/Desktop/tensorFlow/Alphabet_implementation/test_images'
CATEGORIES = ["A", "B", "D", "E", "F", "G", "H", "K", "M", "N", "P", "Q", "R", "T", "V", "Y"]

def skeletonize(img):
    cv = cv2
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv.threshold(img, 127, 255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    done = False
    while (not done):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True
    return skel

def prepare(img):
    IMG_HEIGHT = 50
    imgArr = img
    if (imgArr.shape[1] > 50):
        imgArr = cv2.blur(imgArr, (3,3))
    imgArr = cv2.resize(imgArr, (IMG_HEIGHT, IMG_HEIGHT))
    lowerBound = np.array([20, 100, 130])
    upperBound = np.array([250, 255, 255])
    maskImg = cv2.inRange(imgArr, lowerBound, upperBound)
    #maskImg = skeletonize(maskImg)
    return maskImg.reshape(-1, IMG_HEIGHT, IMG_HEIGHT, 1)


def image_callback(data):
    global filenum
    alpha_image = bridge.imgmsg_to_cv2(data)
    # 92:194
    # 131:328
    print(alpha_image.shape)

    # cropped_image = alpha_image[200:400,340:740]
    cropped_image = alpha_image

    image_large = cropped_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # image_threshold_hsv = cv2.inRange(image, (60,120,180), (255,255,255))
    image_threshold_hsv = cv2.inRange(image, (50,110,170), (255,255,255))
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
            #tempImage = cv2.bitwise_and(image,image,mask=tempMask[:,:,0])
            tempImage = image
            # +20
            img = cv2.getRectSubPix(tempImage,(w+20,h+20),(x+(w/2),y+(h/2)))
            # mask = cv2.getRectSubPix(mask,(w+20,h+20),(x+(w/2),y+(h/2)))
            # sMaskImg = SubImage(x,y,mask)
            sImg = SubImage(x,y,img)
            sub_images.append(sImg)
            prepedImg = np.float32(prepare(cv2.resize(img,(64,64))))
            #prepedImg = np.float32(prepare(os.path.join(testImgDir, 'I/I.194.png')))
            prediction = alphabetModel.predict([prepedImg])
            #print(CATEGORIES[int(prediction[0][0])])
            #print(np.argmax(prediction[0]))

            cv2.circle(image, (x+int(w/2),y+int(h/2)), 1, (0,255,0), thickness=2)
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), thickness=1)
            cv2.putText(image, CATEGORIES[np.argmax(prediction[0])], (x,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=1)
            # Calculate distance
            cameraHeight = 1.27
            cameraFOV = (110/360)*2*math.pi
            realImageWidth = 2*(math.tan(cameraFOV/2)*cameraHeight)
            widthRatio = realImageWidth / img_width

            pythonCommand = "python2.7 send_lcm.py -x " + str(widthRatio*(x-int(img_width/2)+int(w/2))) + " -y " + str(widthRatio*(y-int(img_height/2)+int(h/2))) + " -l " + CATEGORIES[np.argmax(prediction[0])]
            process = subprocess.Popen(pythonCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(CATEGORIES[np.argmax(prediction[0])])
            i+=1
        contouri +=1
    ros_image = bridge.cv2_to_imgmsg(sub_images[0].img)
    ros_image_3 = bridge.cv2_to_imgmsg(sub_images[0].img)
    time.sleep(1)
    for m in sub_images:
        if m is not None:
            # ratioMaskTemp = cv2.inRange(m.img, (15,15,15), (255,255,255))
            #ratioMask = cv2.threshold(ratioMaskTemp, 1, 255, cv2.THRESH_BINARY)
            
            
            # ratioMask = m.img[:,:,2] // m.img[:,:,1]
            # ratioMask2 = m.img[:,:,1] // m.img[:,:,2]
            # kernel = np.ones((5,5),np.uint8)
            # ratioMask = cv2.dilate(ratioMask,kernel)




            # #print(ratioMask)
            # hsv_thresh = cv2.cvtColor(m.img, cv2.COLOR_BGR2HSV)
            # tempImg = cv2.inRange(hsv_thresh, (0,0,100),(255,255,255))
            # m.img = cv2.bitwise_and(m.img,m.img, mask=tempImg)
            # m.img = cv2.cvtColor(m.img, cv2.COLOR_HSV2BGR)

            '''
            ratioTemp = np.divide(m.img[:,:,1],m.img[:,:,2], where=m.img[:,:,2] != 0)
            ratioTemp2 = np.divide(m.img[:,:,2],m.img[:,:,1], where=m.img[:,:,1] != 0)
            ratioTemp *= 255
            ratioTemp2 *= 255
            ratio = ratioTemp.astype(np.uint8)

            ratio = cv2.bitwise_and(ratio, ratio, mask=ratioMask)
            ratio = cv2.bitwise_not(ratio,ratioMask)

            ratio2 = ratioTemp2.astype(np.uint8)
            '''
            # ratio2 = cv2.bitwise_and(ratio2, ratio2, mask=ratioMask2)
            # ratio2 = cv2.bitwise_not(ratio2,cv2.bitwise_not(ratioMask2))

            # threshGreen = cv2.inRange(hsv_thresh,(15,40,10),(70,255,255))
            # threshRed = cv2.inRange(hsv_thresh, (100,175,140),(150,255,255))#, (210,10,10), (255,180,180))

            # ratio = (m.img[:,:,1] / m.img[:,:,2]).astype(np.uint8)

            threshRed = cv2.inRange(m.img, (0,0,220), (50,130,255))
            threshGreen = cv2.inRange(m.img, (0,75,0), (110,255,80))
            ros_image = bridge.cv2_to_imgmsg(m.img)
            #ros_image_3 = bridge.cv2_to_imgmsg(ratio)
            
            
            #divImg = np.divide(m.img[:,:,0],m.img[:,:,1])
            #Blue contour
            xRed,yRed = checkPoint(threshRed)
            #Red contour
            xBlue,yBlue = checkPoint(threshGreen)
            xRed += m.x
            xBlue += m.x
            yRed += m.y
            yBlue += m.y
            # cv2.line(image, (xBlue,yBlue), (xRed, yRed), (0,255,0), thickness=2)

            letter = 'K'
            path = "/home/sa-zhao/perception-python/Kuka-Perception/training_images_v2/" + letter + '/'
            filename = path + letter + str(filenum) + '.png'
            filenum+=1
            #cv2.imwrite(filename,m.img)
            #time.sleep(0.02)
            #print(filenum)
    print("Write")
    ros_image2 = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)
    image_pub2.publish(ros_image2)
    # image_pub3.publish(ros_image_3)
    # filename = 'TestImage' + str(filenum) + '.png'
    filename = 'crop_test.png'
    #filenum+=1
    cv2.imwrite(filename,image)


rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, image_callback)

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

if __name__ == "__main__":
    rospy.spin()