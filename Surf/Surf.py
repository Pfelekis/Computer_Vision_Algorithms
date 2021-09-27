from __future__ import print_function
import cv2 
import numpy as np
import argparse

src = cv2.imread('faros.png', cv2.IMREAD_GRAYSCALE)

#-- Step 1: Detect the keypoints using SURF Detector
minHessian = 5000
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints ,des = detector.detectAndCompute(src,None)
#-- Draw keypoints
#img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
img=cv2.drawKeypoints(src, keypoints,None,(255,0,0),4)
#-- Show detected (drawn) keypoints
cv2.imshow('SURF Keypoints', img)
cv2.imwrite('Surf_keypoints.jpg',img)
cv2.waitKey()
