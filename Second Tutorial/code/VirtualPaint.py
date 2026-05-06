import cv2
import numpy as np
framewidth=640
frameheight=480

cap=cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)
cap.set(10,150)

# myColors=[[65,89,72,86,202,217]]
myColors=[[50,40,195,80,89,221]]
myColorValues=[[0,204,0]]
myPoints=[]

def FindColor(img,myColors,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count=0
    newPoints=[]
    lower=np.array(myColors[0][0:3])
    upper=np.array(myColors[0][3:6])
    mask=cv2.inRange(imgHSV,lower,upper)
    x,y=getContours(mask,myColorValues)
    cv2.circle(imgResult,(x,y),10,myColorValues[0],cv2.FILLED)
    if x!=0 and y!=0:
        newPoints.append([x,y,count])
    cv2.imshow('img',mask)
    return newPoints

def getContours(img,myV):
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

    x,y,w,h=0,0,0,0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgResult,[cnt],-1,myV[0],10)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
    return (x+x+w)//2,(y+y+h)//2

def drawOnCanvas(myPoints,myColorValues,):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,myColorValues[point[2]],cv2.FILLED) 

           

while True:
    success,img=cap.read()
    if success:
        imgResult=img.copy()
        newPoints=FindColor(img,myColors,myColorValues)
        if len(newPoints)!=0:
            for newP in newPoints:
                myPoints.append(newP)
        if len(myPoints)!=0:
            drawOnCanvas(myPoints,myColorValues)
        cv2.imshow('Res',img)
        cv2.imshow('rs',imgResult)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break