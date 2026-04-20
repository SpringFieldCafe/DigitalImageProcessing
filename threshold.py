import cv2 as c

gray=c.imread("graph\\bookpage.jpg",c.IMREAD_GRAYSCALE)
ret,binary=c.threshold(gray,10,255,c.THRESH_BINARY)
binary_adaptive=c.adaptiveThreshold(gray,255,c.ADAPTIVE_THRESH_GAUSSIAN_C,c.THRESH_BINARY,115,1)
ret1,binary_otsu=c.threshold(gray,0,255,c.THRESH_BINARY+c.THRESH_OTSU)


c.imshow("g",gray)
c.imshow("b",binary)
c.imshow("ba",binary_adaptive)
c.imshow("otsu",binary_otsu)

c.waitKey()