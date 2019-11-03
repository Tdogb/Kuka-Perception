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

img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0
distanceFromGround = 1

zed = sl.Camera()

def image_callback(src_image, src_depth):
    global filenum
    image_large = src_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    # hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #rangeImage = cv2.inRange(hsv, (0,10,10),(360,255,255))
#    maskedImage = cv2.bitwise_and(rangeImage, image)
    depth_mask = np.empty_like(image_large)
    lowerIndices = src_depth <= 3
    upperIndices = src_depth > 3
    depth_mask[lowerIndices] = 255
    depth_mask[upperIndices] = 0 
     
    grayscale = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(       grayscale,100,200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ret, edges = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) > 30:
            cv2.drawContours(image, contours, i, (0,255,0), 2)
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(depth_mask,cmap = 'gray')
    plt.show()
    filename = 'I.' + str(filenum) + '.png'
    filenum+=1  

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
        tempArray = np.vstack((np.linspace(inputArray[i,0],inputArray[i+1,0],points_at_each_segment, endpoint=False),np.linspace(inputArray[i,1],inputArray[i+1,1],points_at_each_segment, endpoint=False)))
        row0 = (np.append(output[0,:],tempArray[0,:])).T
        row1 = (np.append(output[1,:],tempArray[1,:])).T
        output = np.vstack((row0,row1))
    print(output.shape)
    print("Desired Size")
    print(desiredSize)
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
        image_callback(image_zed.get_data(), depth_zed.get_data())
    else:
        print("Failed")

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
        zed.close()
        key = cv2.waitKey(10)

if __name__ == "__main__":
    zed.close()
    initCamera()
    main()