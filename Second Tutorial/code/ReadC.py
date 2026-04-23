import cv2

video=cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
video.set(10,70)

while 1:
    success,img=video.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1)& 0xFF ==ord('q'):
        break