import cv2 as cv

img=cv.imread("graph\\opencv_logo.jpg")


cv.imshow("blue",img[:,:,0])
cv.imshow('green',img[:,:,1])
cv.imshow('red',img[:,:,2])

cv.waitKey()

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

cv.waitKey()