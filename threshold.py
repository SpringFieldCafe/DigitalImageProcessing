import cv2 as c

gray=c.imread("graph\\bookpage.jpg",c.IMREAD_GRAYSCALE)
ret,binary=c.threshold(gray,10,255,c.THRESH_BINARY)

c.imshow("g",gray)
c.imshow("b",binary)

c.waitKey()