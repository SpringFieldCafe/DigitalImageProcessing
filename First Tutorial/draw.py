import numpy as np
import cv2 as c

img=np.zeros([300,300,3],dtype=np.uint8)
ig = np.random.randint(0, 256, size=(800, 800, 3), dtype=np.uint8)

c.imshow('ig',ig)
c.line(img,(100,200),(250,250),(250,0,0),2)

c.rectangle(img,(30,100),(60,150),(0,250,0),7)
c.circle(img,(150,100),20,(0,0,250),3)
c.putText(img,'hello',(100,50),0,1,(200,200,200),2,1)
c.imshow("img",img)
c.waitKey()