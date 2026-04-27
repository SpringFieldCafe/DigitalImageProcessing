import cv2
import numpy as np
import matplotlib.pyplot as plt
from Modal_Stack import stackImages

def empty(x):
    pass

# def draw_3d(X, Y, Z, title, zlabel):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot_surface(X, Y, Z)

#     ax.set_title(title)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel(zlabel)

#     # 让 Y 轴方向更接近图像坐标系
#     ax.invert_yaxis()

path = "Second Tutorial\\resources\\lambo.PNG"
h_min = 0
h_max = 19
s_min = 110
s_max = 240
v_min = 153
v_max = 255

while 1:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)
    imgRes=cv2.bitwise_and(img,img,mask=mask)

    imgStack=stackImages(0.8,[[img,imgHSV],[imgRes,mask]])
    cv2.imshow('s',imgStack)
    cv2.waitKey(1)