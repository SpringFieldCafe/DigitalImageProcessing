import cv2


faceCascade=cv2.CascadeClassifier("Second Tutorial\\resources\\haarcascade_frontalface_default.xml")
img=cv2.imread('Second Tutorial\\resources\\lena.png')
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(180,100,30),4)
    

cv2.imshow('Res',img)
cv2.waitKey()