import cv2
import numpy as np
kernel=np.ones((5,5),dtype=np.uint8)

img=cv2.imread('Second Tutorial\\resources\\lena.png',cv2.IMREAD_GRAYSCALE)
canny=cv2.Canny(img,100,200)
imgdialation=cv2.dilate(canny,kernel,iterations=1)
imgeroded=cv2.erode(imgdialation,kernel,iterations=2)

cv2.imshow('img',img)
cv2.imshow('dia',imgdialation)
cv2.imshow('erosion',imgeroded)
cv2.waitKey()