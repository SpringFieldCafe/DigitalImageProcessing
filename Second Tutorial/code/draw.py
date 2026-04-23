import cv2
import numpy as np


print(__name__)
bg=np.zeros((400,640,3),dtype=np.uint8)
white=np.ones((300,400,3),dtype=np.uint8)*122
rom=np.random.randint(0,256,(300,300),dtype=np.uint8)
rom[::]=122
cv2.line(bg,(30,20),(bg.shape[1],bg.shape[0]),(255,0,255),4)
## cv2.rectangle(bg,(10,10),(250,350),(0,0,255),cv2.FILLED)
cv2.rectangle(bg,(30,100),(200,200),(0,255,0),3)
cv2.circle(bg,(400,50),30,(245,182,47),5)
cv2.putText(bg,"open CV",(300,100),cv2.FONT_HERSHEY_COMPLEX,.3,(120,250,180))

cv2.imshow('bg',bg)
cv2.imshow('wh',white)
cv2.imshow('rom',rom)
cv2.waitKey()