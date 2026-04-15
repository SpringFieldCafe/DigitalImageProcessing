import cv2

img=cv2.imread("graph\\opencv_logo.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners=cv2.goodFeaturesToTrack(gray,500,0.1,10)

for corner in corners:
    x,y=corner.ravel()
    cv2.circle(img,(int(x),int(y)),3,(255,0,255),-1)

cv2.imshow("corners",img)
cv2.waitKey()