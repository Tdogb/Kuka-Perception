# Kuka Perception Algorithms
## High-level Goals:
* Simple
* Robust and reliable performance
* Able to classify generic objects given a fixed reference plane (e.g a table)

## High-level algorithm overview
1. Reset the camera fused pointcloud at a known arm pose
2. Place a reference plane on where the table is expected to be
    * Method 1
        1. Determine if the error of the least squares regression plane is within tolerance
        2. Check the bounds of the edge of the table
    * Method 2
        1. Calculate the least squares regression plane through the table
        2. Set this as the reference plane rather than using a prediction
3. Remove all data in the pointcloud below and on this plane.
4. Use RGB data from the camera and the projection of this plane onto the RGB image to segment/mask the objects
5. Mask the fused pointcloud (by calculating the intercection of all masks)
6. Run ICP with noise rejection

## Low-level Goals:
* Retain a classification of the objects throughout camera translation/rotation

## Structure:
* Receive frames from camera
* Once signal is recived that arm is in desired spot, run zed.enableSpatialMapping() and enable tracking
* Run findFloorPlane()