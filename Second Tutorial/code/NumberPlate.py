import cv2

framewidth=640
frameheight=480
minArea=500
npCascade=cv2.CascadeClassifier("Second Tutorial\\resources\\haarcascade_russian_plate_number.xml")
color=(120,160,240)
count=0

cap=cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)
cap.set(10,150)
while True:
    success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates=npCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in numberPlates:
        area=w*h
        if area>minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,4)
            cv2.putText(img,"number plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            imgRoi=img[y:y+h,x:x+w]
            cv2.imshow('Roi',imgRoi)
    
    cv2.imshow("res",img)
    if cv2.waitKey(1)& 0xFF==ord('s'):
        cv2.imwrite("Second Tutorial\\resources\\Scanned\\pla_"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),color,cv2.FILLED)
        cv2.putText(img,"ScanSaved",(150,265),cv2.FONR_HERSHEY_DUPLEX,2,(0,0,255),2)
        cv2.imshow("RS",img)
        cv2.waitKey(1000)
        count+=1
