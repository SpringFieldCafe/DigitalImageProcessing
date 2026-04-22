import cv2

img=cv2.imread('Second Tutorial\\resources\\lena.png')
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny=cv2.Canny(img,100,200)

cv2.imshow("img",imgGray)
cv2.imshow("blur",imgBlur)
cv2.imshow('canny',imgCanny)

cv2.waitKey()