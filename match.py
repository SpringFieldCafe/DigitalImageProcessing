import cv2 as c
import numpy as np

img=c.imread("graph\\poker.jpg")
gray=c.cvtColor(img,c.COLOR_BGR2GRAY)

template=gray[75:105,235:265]

match=c.matchTemplate(gray,template,c.TM_CCOEFF_NORMED)
loc=np.where(match>=0.9)

h,w=template.shape[0:2]
for p in zip(*loc[::-1]):
    x1,y1=p[0],p[1]
    x2,y2=x1+w,y1+h
    c.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)


c.imshow('img',img)
c.waitKey()