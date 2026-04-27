import cv2
import numpy as np
import matplotlib.pyplot as plt
from Modal_Stack import stackImages as si

def empty(x):
    pass

def draw_3d(X, Y, Z, title, zlabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(zlabel)

    # 让 Y 轴方向更接近图像坐标系
    ax.invert_yaxis()

path = "Second Tutorial\\resources\\lambo.PNG"
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while 1:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min=cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max=cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min=cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max=cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min=cv2.getTrackbarPos("Val Min","TrackBars")
    v_max=cv2.getTrackbarPos("Val Max","TrackBars")
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)
    imgRes=cv2.bitwise_and(img,img,mask=mask)
    Is=si(0.6,[[img,imgHSV],[mask,imgRes]])
    cv2.imshow("s",Is)
    cv2.waitKey(1)




print(imgHSV)
print(imgHSV.shape)

# 分离 H、S、V 三个通道
H = imgHSV[:, :, 0]
S = imgHSV[:, :, 1]
V = imgHSV[:, :, 2]

# 获取图像尺寸
height, width, channels = imgHSV.shape

# 生成 X、Y 坐标
X, Y = np.meshgrid(np.arange(width), np.arange(height))

# 图像像素太多，直接画会很卡，所以间隔采样
step = 5

X_sample = X[::step, ::step]
Y_sample = Y[::step, ::step]
H_sample = H[::step, ::step]
S_sample = S[::step, ::step]
V_sample = V[::step, ::step]

cv2.imshow("original", img)
cv2.imshow("HSV", imgHSV)

draw_3d(X_sample, Y_sample, H_sample, "3D Visualization of H Channel", "H")
draw_3d(X_sample, Y_sample, S_sample, "3D Visualization of S Channel", "S")
draw_3d(X_sample, Y_sample, V_sample, "3D Visualization of V Channel", "V")

plt.show()
cv2.waitKey()


cv2.destroyAllWindows()