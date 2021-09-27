import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('faros.jpg',0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow("ORB",img2)
cv.imwrite('ORB_features.jpg',img2)