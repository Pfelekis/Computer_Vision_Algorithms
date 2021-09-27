import cv2 
import numpy as np




img= cv2.imread("cat.png",1)

img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.imwrite('GrayscaleCat.png', img)

img= cv2.GaussianBlur(img, (5,5), 1.4)


#cv2.imwrite('GaussianCat.png', img)
edges = cv2.Canny(img,50,200)

cv2.imwrite('CannyCat.png', edges)