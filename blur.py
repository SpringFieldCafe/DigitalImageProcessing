import cv2 as c

img=c.imread("graph/plane.jpg")


gauss=c.GaussianBlur(img,(5,5),0)
median=c.medianBlur(img,5)


c.imshow("img",img)
c.imshow("Gauss",gauss)
c.imshow('median',median)
c.waitKey()