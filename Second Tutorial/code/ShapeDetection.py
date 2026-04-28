import cv2
import numpy as np
from Modal_Stack import stackImages as s


def getContours(img):
    contours_info = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 兼容 OpenCV 3 和 OpenCV 4
    if len(contours_info) == 3:
        _, contours, hierarchy = contours_info
    else:
        contours, hierarchy = contours_info
    for cnt in contours:
        area=cv2.contourArea(cnt)
        print(area)
        if area>500:
            vb=cv2.drawContours(imgContour,[cnt],-1,(180,90,30),10)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            print(approx)
            objCor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)

            if objCor==3:
                objectType="Tri"
            elif objCor==4:
                aspRation=w/float(h)
                if aspRation>0.95 and aspRation<1.05:
                    objectType="Squ"
                else:
                    objectType='Rec'
            elif objCor>4:
                objectType='cir'
            else:
                objectType="None"
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),4)
            cv2.putText(imgContour,objectType
                        ,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,
                        (0,0,0),2)
    return vb

path='Second Tutorial\\resources\\shapes.png'
img=cv2.imread(path)
imgContour=img.copy()

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)
imgBlank=np.zeros_like(img)
v=getContours(imgCanny)

imgStack=s(0.4,[[imgGray,img,imgContour],[imgCanny,imgBlur,v]])
cv2.imshow('st',imgStack)
cv2.waitKey(0)