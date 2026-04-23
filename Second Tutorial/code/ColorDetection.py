import cv2
import numpy as np

path="Second Tutorial\\resources\\lambo.PNG"
img=cv2.imread(path)

imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.imshow("original",img)
cv2.imshow("HSV",imgHSV)
cv2.waitKey()