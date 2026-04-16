import cv2 as c
import numpy as np

img=c.imread("graph\\poker.jpg")
gray=c.cvtColor(img,c.COLOR_BGR2GRAY)

template=gray[75:105,235:265]
c.imshow("temp",template)

threshold=0.9
match=c.matchTemplate(gray,template,c.TM_CCOEFF_NORMED)
loc=np.where(match>=threshold)

print("!",loc)
print("1",loc[::-1])
print("1",*loc[::-1])
print(zip(*loc[::-1]))

h,w=template.shape[0:2]
print(template.shape)
print(type(template.shape))
print((3,4,5,8)[2])
c1=0
for p in zip(*loc[::-1]):
    c1+=1
    x1,y1=p[0],p[1]
    x2,y2=x1+w,y1+h
    c.rectangle(img,(x1,y1),(x2,y2),(100,100,100),2,)


print(c1)
c.imshow('img',img)
c.waitKey()