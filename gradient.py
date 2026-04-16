import cv2 as c

gray=c.imread("graph\\opencv_logo.jpg",c.IMREAD_GRAYSCALE)

laplacian=c.Laplacian(gray,c.CV_64F)
canny=c.Canny(gray,100,200)


c.imshow("g",gray)
c.imshow("la",laplacian)
c.imshow("canny",canny)

c.waitKey()