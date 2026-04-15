import cv2 as cv

print(cv.getVersionString())

image=cv.imread("graph/opencv_logo.jpg")

print(image.shape)

cv.imshow("image",image)