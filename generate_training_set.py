import sys
import skimage
from skimage.transform import resize, rotate
from skimage.util import random_noise, crop
import skimage.viewer
import numpy as np
import math
import time

path = "/home/sa-zhao/perception-python/Kuka-Perception/images/IMG_9940.JPG"
initial_image = skimage.io.imread(path)
image = resize(initial_image, (254,254))


for theta in range(0, 360):
    rotated_image = rotate(image, theta)
    random_noise(rotated_image)
    viewer = skimage.viewer.ImageViewer(rotated_image)
    viewer.show()
    time.sleep(0.5)