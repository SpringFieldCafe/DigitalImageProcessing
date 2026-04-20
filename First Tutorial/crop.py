import cv2 

img=cv2.imread("graph/opencv_logo.jpg")

crop=img[10:170,40:200]

cv2.imshow("crop",crop)
cv2.waitKey()