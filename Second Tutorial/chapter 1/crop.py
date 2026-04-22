import cv2
import numpy as np


img=cv2.imread('Second Tutorial\\resources\\lambo.PNG')

print(img.shape)
crop=img[10:400,100:600]
imgResize=cv2.resize(img,(300,200))

cv2.imshow('size',imgResize)
cv2.imshow('ceop',crop)

cv2.waitKey()

