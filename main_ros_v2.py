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
    grayscale = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grayscale,100,200)

    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.show()

    ros_image = bridge.cv2_to_imgmsg(edges)
    ros_image2 = bridge.cv2_to_imgmsg(image)
    image_pub.publish(ros_image)
    image_pub2.publish(ros_image2)
    filename = 'I.' + str(filenum) + '.png'
    filenum+=1  
    #cv2.imwrite(filename,sub_images[0])

def depth_callback(data):
    if false:
        continue

rospy.init_node("perception_node")
image_sub = rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, image_callback)
depth_sub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_callback)

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


if __name__ == "__main__":
    rospy.spin()