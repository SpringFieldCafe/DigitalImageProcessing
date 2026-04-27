import cv2
import numpy as np
from Modal_Stack import stackImages as s


def getContours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)






path='Second Tutorial\\resources\\shapes.png'
img=cv2.imread(path)

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)
imgBlank=np.zeros_like(img)

imgStack=s(0.7,[[imgGray,img,imgBlank],[imgCanny,imgBlur]])
cv2.imshow('st',imgStack)
cv2.waitKey(0)