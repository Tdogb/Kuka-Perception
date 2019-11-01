import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from sklearn.neighbors import NearestNeighbors
from cv_bridge import CvBridge, CvBridgeError
import math
import matplotlib.pyplot as plt
import pdb

bridge = CvBridge()
image_pub = rospy.Publisher("image_publisher_zed_camera", Image, queue_size=10)
image_pub2 = rospy.Publisher("image_2", Image, queue_size=10)

img_width = 640
img_height = 360

upper_color_bounds = np.array([255,255,255])
lower_color_bounds = np.array([75,75,100])

filenum = 0

def image_callback(data):
    global filenum
    alpha_image = bridge.imgmsg_to_cv2(data)
    image_large = alpha_image[:,:,:3]
    image = cv2.resize(image_large, (img_width,img_height))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image_threshold_hsv = cv2.inRange(image, (0,61,98), (33,255,255))
    image_threshold_hsv = cv2.inRange(image, (0,61,98), (33,255,255))
    image_oshu = cv2.threshold(grayscale, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    contour_image = np.uint8(image_threshold_hsv)
    contours, heirarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sub_images = [None] * len(contours)
    i = 0
    contouri = 0
    output_contour_index = 0
    for c in contours:
        if cv2.contourArea(contours[contouri]) > 30:
            output_contour_index = contouri
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            if w > h:
                h = w
            if h > w:
                w = h
            sub_images[i] = cv2.getRectSubPix(image,(w+20,h+20),(x+(w/2),y+(h/2)))
            sub_images[i] = cv2.resize(sub_images[i], (64,64))
            #cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1
            i+=1
        contouri +=1
    # Begin ICP
    # tMesh = np.array([[-4,-23],[-4,18],[-17,18],[-17,25.3], [4,-23],[4,18],[17,18],[17,25.3]])
    tMesh = np.array([[-4,-23],[-4,18],[-17,18],[-17,25.3], [17,25.3],[17,18],[4,18],[4,-23]])
    rotationMatrix = np.matrix([[0,-1],[1,0]])
    tMeshMat = np.asmatrix(tMesh).T
    tRotatedMat = np.empty_like(tMeshMat)
    for i in range(0, np.size(tMeshMat,1)):
        tRotatedMat[:,i] = rotationMatrix * tMeshMat[:,i]
    tRotatedMeshArr = np.asarray(tRotatedMat.T)
    #print(tMesh)
    #print(tRotatedMeshArr)
    #print(" ")
    testMesh = np.array([[0,0],[2,1],[3,2],[4,4]])
    #generateRefMesh(testMesh, 4, 16)
    #print(contours[output_contour_index].shape[0])

    #a,b,c = best_fit_transform(tMesh, tRotatedMeshArr)
    #print(b)
    # End ICP

    print("Contours shape")
    print(contours[output_contour_index].shape)
    a,b,c = best_fit_transform(np.squeeze(contours[output_contour_index]),generateRefMesh(tMesh, contours[output_contour_index].shape[0]))
    print(b)
    vec = np.matrix([[0],[1]])
    movedVec = np.squeeze(10*(b*vec))
    print("Moved vec")
    print(movedVec[0,1])
    first = math.ceil(movedVec[0,0])+200
    second = math.ceil(movedVec[0,1])+200
    print(first)
    print(second)
    cv2.arrowedLine(image, (200,200), (first,second), (0,255,0))

    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    ros_image = bridge.cv2_to_imgmsg(image_oshu)
    ros_image2 = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)
    image_pub2.publish(ros_image2)
    filename = 'I.' + str(filenum) + '.png'
    filenum+=1  
    #cv2.imwrite(filename,sub_images[0])

    # print("tshape")
    # print("Contour shape")
    # print(contours.shape)
    #print(contours[output_contour_index])
    # print(tMesh.shape[0])


rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, image_callback)

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