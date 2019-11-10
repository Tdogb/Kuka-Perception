import sys
import time
import rospy
import cv2
import math
import pdb
import numpy as np
import pyzed.sl as sl
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

usingZed = False

img_width = 640 if usingZed else 1280
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0
distanceFromGround = 1

zed = sl.Camera()
fig = plt.figure(1)
fig2 = plt.figure(2)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax3 = fig2.add_subplot(121)

tempMesh = np.array([[-4,-23],[-4,18],[-17,18],[-17,25.3], [17,25.3],[17,18],[4,18],[4,-23]])
ax3.set_xlim(0,200)
ax3.set_ylim(500,700)
#sc1 = ax3.scatter(tempMesh[:,0],tempMesh[:,1])


tempImage = np.zeros((img_height,img_width,3))

im1 = ax1.imshow(tempImage,cmap = 'gray')
im2 = ax2.imshow(tempImage,cmap = 'gray')
plt.ion()

def image_callback(src_image):
    global filenum
    image_large = src_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))

    grayscale = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grayscale,80,200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    indicies = []
    #tMesh = np.array([[-4,-23],[-4,18],[-17,18],[-17,25.3], [17,25.3],[17,18],[4,18],[4,-23]])

    sub_images = [None] * len(contours)
    sub_masks = [None] * len(contours)
    i = 0
    contouri = 0
    #print(contours)
    for c in contours:
        if cv2.contourArea(contours[contouri]) > 1000:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            if w > h:
                h = w
            if h > w:
                w = h
            sub_images[i] = cv2.getRectSubPix(grayscale,(np.int32(w)+20,np.int32(h)+20),(x+(np.int32(w)/2),y+(np.int32(h)/2)))
            sub_images[i] = cv2.resize(sub_images[i], (64,64))
            sub_masks[i] = cv2.threshold(sub_images[i],0,100,cv2.THRESH_BINARY)
            i+=1
        contouri +=1

    # Initiate SIFT detector
    #sift = cv2.SIFT()
    img1 = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/FLANN_Test_Image.png",0)
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    #imgGray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    kp1 = orb.detect(img1,None)
    kp2 = orb.detect(sub_images[0],None)
    kp1, des1 = orb.compute(img1,kp1)
    kp2, des2 = orb.compute(sub_images[0],kp2)


    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,sub_images[0],kp2,matches,None,**draw_params)
    print("Start image printing")
    print(im1)
    print("sub images")
    print(sub_images[0])
    im1.set_data(im1) 
    im2.set_data(sub_images[0])  

def receiveData():
    image_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
    depth_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_32F_C1)
    
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    while zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        pass
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.MEASURE_DEPTH)
        # Use get_data() to get the numpy array
        image_callback(image_zed.get_data())
    # else:
    #     print("Failed")

def initCamera():
    # Create a ZED camera object
    # Set configuration parameters
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

def main():
    key = ' '
    while key != 123:
        receiveData()
        #zed.close()
        plt.pause(0.001)
        #key = cv2.waitKey(10)

def staticImageMain():
    img = cv2.imread("/home/sa-zhao/perception-python/Kuka-Perception/TImage.png")
    image_callback(img)

if __name__ == "__main__":
    if usingZed:
        zed.close()
        initCamera()
        main()
    else:
        staticImageMain()
    plt.ioff()
    plt.show()