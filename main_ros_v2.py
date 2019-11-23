import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud, PointCloud2
# from sklearn.neighbors import NearestNeighbors
from cv_bridge import CvBridge, CvBridgeError
import math
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from tensorflow import keras

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)
image_pub2 = rospy.Publisher("image_2", Image, queue_size=10)

img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0

alphabetModel = keras.models.load_model('alphabetTrained26.h5')
print("Finished loaded trained weights")


testImgDir = '/home/tannerliu/Desktop/tensorFlow/Alphabet_implementation/test_images'
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]



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
    alpha_image = bridge.imgmsg_to_cv2(data)
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
    ros_image = bridge.cv2_to_imgmsg(sub_images[0].img)
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
            ros_image = bridge.cv2_to_imgmsg(ratio)
            
            
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


    ros_image2 = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)
    image_pub2.publish(ros_image2)
    # filename = 'TestImage' + str(filenum) + '.png'
    filename = 'TestDots2.png'
    filenum+=1  
    # cv2.imwrite(filename,image)


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

def generateRefMesh(inputArray, desiredSize):
    arraySize = inputArray.shape[0]
    points_at_each_segment = math.floor(desiredSize/arraySize)
    points_at_first_segment = desiredSize - (points_at_each_segment*(arraySize-1))
    output = np.zeros((2,points_at_first_segment))
    # print(points_at_each_segment)
    # print(points_at_first_segment)
    #print(output)
    output[0,:] = (np.linspace(inputArray[0,0],inputArray[1,0],points_at_first_segment))
    output[1,:] = (np.linspace(inputArray[0,1],inputArray[1,1],points_at_first_segment))
    #print(output)

    for i in range(0, arraySize-1):
        # print("Linspace")
        # print(np.linspace(inputArray[i,0],inputArray[i+1,0],points_at_each_segment,endpoint=False))
        tempArray = np.vstack((np.linspace(inputArray[i,0],inputArray[i+1,0],points_at_each_segment, endpoint=False),np.linspace(inputArray[i,1],inputArray[i+1,1],points_at_each_segment, endpoint=False)))
        row0 = (np.append(output[0,:],tempArray[0,:])).T
        row1 = (np.append(output[1,:],tempArray[1,:])).T
        output = np.vstack((row0,row1))
    #     print("FirstOutput")
    #     print(output)
    #     print("FirstOutput Part")
    #     print(output[0,:])
    #     print("TempArray")
    #     print(tempArray)
    #     print("TempArray Part")
    #     print(tempArray[0,:])
    #     print("Row 0: ")
    #     print(row0)
    #     print("Row 1")
    #     print(row1)
    #     print("Output")
    #     print(output)
    #     print("Done")
    # print(output)
    print(output.shape)
    print("Desired Size")
    print(desiredSize)
    # print("First Step size")
    # print(points_at_first_segment)
    # print("Normal step size")
    # print(points_at_each_segment)
    # print(inputArray.shape)
    # plt.scatter(output.T[:,0],output.T[:,1])
    # plt.show()
    return output.T

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape 
    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


if __name__ == "__main__":
    rospy.spin()