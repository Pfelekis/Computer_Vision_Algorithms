"""A simple Sobel-Feldman edge detector. I simple multiply the image with two parameters:

   [-1  0  1       [-1 -2 -1
Gx= -2  0  2     Gy= 0  0  0
    -1  0  1]        1  2  1]

"""

import cv2
import numpy as np 

#read the image in greyscale
img = cv2.imread('cat.png', 0)

#we will use sobel operator for this task

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

#detecting horizontal edges
hor_img = cv2.addWeighted(sobelx, 0, sobely, 1, 0)
cv2.imwrite('hor_edge.jpg', hor_img)


#detecting horizontal edges
ver_img = cv2.addWeighted(sobelx, 1, sobely, 0, 0)
cv2.imwrite('ver_edge.jpg', ver_img)

sobelxy_img=cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imwrite('soblexy_img.jpg',sobelxy_img)