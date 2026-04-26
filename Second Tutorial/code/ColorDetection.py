import cv2
import numpy as np
import matplotlib.pyplot as plt


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

img = cv2.imread(path)

if img is None:
    print("图片读取失败，请检查路径")
    exit()

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(imgHSV)
print(imgHSV.shape)

cv2.imshow("original", img)
cv2.imshow("HSV", imgHSV)
cv2.waitKey(1)

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
cv2.waitKey(1)


cv2.destroyAllWindows()