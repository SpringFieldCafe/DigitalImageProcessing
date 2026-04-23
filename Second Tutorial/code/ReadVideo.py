import cv2

video=cv2.VideoCapture("Second Tutorial\\resources\\test_video.mp4")


#go through each frame

while 1:
    success,img=video.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break