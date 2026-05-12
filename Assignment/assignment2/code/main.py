import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import os

code_dir=Path(__file__).resolve().parent
assignment2_dir=code_dir.resolve().parent
src_dir=assignment2_dir.resolve()/"resource"
g41_path=src_dir/"g41_0.jpg"
miku_path=src_dir/"miku_0.jpg"

def stackImages(scale,imgArray):
    rowsAvailable=isinstance(imgArray[0],list)
    if rowsAvailable:
        width=imgArray[0][0].shape[1]
        height=imgArray[0][0].shape[0]
        rows=len(imgArray)
        cols= max(len(row) for row in imgArray)
        imgBlack = np.zeros_like(imgArray[0][0])
        for x in range(0,rows):
            while len(imgArray[x]) < cols:
                imgArray[x].append(imgBlack.copy())
            for y in range(0,cols):
                if imgArray[x][y].shape[:2]==imgArray[0][0].shape[:2]:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                else:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)
                if len(imgArray[x][y].shape)==2:
                    imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imgBlack=np.zeros((height,width,3),np.uint8)
        hor=[imgBlack]*rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])
        ver=np.vstack(hor)
    else:
        width=imgArray[0].shape[1]
        height=imgArray[0].shape[0]
        cols=len(imgArray)

        for x in range(0,cols):
            if imgArray[x].shape[:2]==imgArray[0].shape[:2]:
                imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x]=cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)  
            if len(imgArray[x].shape)==2:
                imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        ver=hor
    return ver


#######################################
#1
imgG41=cv2.imread(str(g41_path))
imgG41_hsv = cv2.cvtColor(imgG41, cv2.COLOR_BGR2HSV)  # BGR 转 HSV 色彩空间
imgG41_hls = cv2.cvtColor(imgG41, cv2.COLOR_BGR2HLS)  # BGR 转 HLS 色彩空间
imgG41_lab = cv2.cvtColor(imgG41, cv2.COLOR_BGR2LAB)  # BGR 转 LAB 色彩空间
imgG41_yuv = cv2.cvtColor(imgG41, cv2.COLOR_BGR2YUV)  # BGR 转 YUV 色彩空间
imgG41stack=stackImages(0.3,[[imgG41_hls,imgG41_hsv],
                             [imgG41_lab,imgG41_yuv]])
cv2.imshow('stack',imgG41stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
########################################################


###########################################
#2
imgG41_gray = cv2.cvtColor(imgG41, cv2.COLOR_BGR2GRAY)  # 将原始 BGR 图像转换为灰度图像
k1 = 1.2  # 设置线性变换系数 K=1.2，用于增强灰度图像对比度
k2 = 0.7  # 设置线性变换系数 K=0.7，用于减弱灰度图像对比度
b = 0  # 设置线性变换偏置 b=0
imgG41_gray_float = imgG41_gray.astype(np.float32)  # 将灰度图像转换为 float32 类型，避免计算时发生溢出
imgG41_gray_enhance = k1 * imgG41_gray_float + b  # 根据公式 g(x,y)=K*f(x,y)+b 进行对比度增强变换
imgG41_gray_weaken = k2 * imgG41_gray_float + b  # 根据公式 g(x,y)=K*f(x,y)+b 进行对比度减弱变换
imgG41_gray_enhance = np.clip(imgG41_gray_enhance, 0, 255)  # 将增强后的像素值限制在 0 到 255 之间
imgG41_gray_weaken = np.clip(imgG41_gray_weaken, 0, 255)  # 将减弱后的像素值限制在 0 到 255 之间
imgG41_gray_enhance = imgG41_gray_enhance.astype(np.uint8)  # 将增强后的图像转换回 uint8 类型，便于显示和保存
imgG41_gray_weaken = imgG41_gray_weaken.astype(np.uint8)  # 将减弱后的图像转换回 uint8 类型，便于显示和保存
imgG41_gray_stack = stackImages(0.5, [[imgG41_gray, imgG41_gray_enhance, imgG41_gray_weaken]])  # 将原灰度图、增强图、减弱图拼接显示
cv2.imshow("gray_linear_transform", imgG41_gray_stack)  # 显示灰度线性变换结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
########################################################

#################################################################
#3
imgG41_gray_float = imgG41_gray.astype(np.float32)  # 将灰度图像转换为 float32 类型，便于进行对数运算
c = 255 / np.log(1 + np.max(imgG41_gray_float))  # 根据最大灰度值计算对数变换系数 c，使结果范围映射到 0 到 255
imgG41_log = c * np.log(1 + imgG41_gray_float)  # 根据公式 g(x,y)=c*log(1+f(x,y)) 进行对数非线性变换
imgG41_log = np.clip(imgG41_log, 0, 255)  # 将变换后的像素值限制在 0 到 255 之间
imgG41_log = imgG41_log.astype(np.uint8)  # 将图像数据转换回 uint8 类型，便于显示和保存
imgG41_log_stack = stackImages(0.5, [[imgG41_gray, imgG41_log]])  # 将原灰度图和对数变换结果图拼接显示
cv2.imshow("gray_log_transform", imgG41_log_stack)  # 显示灰度图像对数变换结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
###################################################

###########################################
#4
# 图像直方图是统计图像中各个灰度级像素出现次数或频率的图形，它可以反映图像灰度分布情况
# 直方图均衡化是一种灰度变换方法，通过重新分配灰度级，使图像灰度分布更加均匀
# 对图像进行直方图均衡化后，通常可以增强图像对比度，使暗区和亮区的细节更加明显
# cv2.equalizeHist() 函数只能直接用于单通道灰度图像，不能直接用于三通道彩色图像
imgG41_gray = cv2.cvtColor(imgG41, cv2.COLOR_BGR2GRAY)  # 将原始 BGR 图像转换为灰度图像
imgG41_equal = cv2.equalizeHist(imgG41_gray)  # 使用 cv2.equalizeHist() 对灰度图像进行直方图均衡化
imgG41_hist_stack = stackImages(0.5, [[imgG41_gray, imgG41_equal]])  # 将原灰度图和直方图均衡化后的图像拼接显示
cv2.imshow("gray_hist_equalization", imgG41_hist_stack)  # 显示灰度图像直方图均衡化前后的对比结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
####################################################################


###########################################
#5
imgG41_rgb = cv2.cvtColor(imgG41, cv2.COLOR_BGR2RGB)  # 将 OpenCV 默认读取的 BGR 图像转换为 RGB 图像，便于按 RGB 通道绘制直方图
imgG41_gray = cv2.cvtColor(imgG41, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为空间变换后的灰度图像
R, G, B = cv2.split(imgG41_rgb)  # 将 RGB 图像分离为 R、G、B 三个颜色通道
hist_R = cv2.calcHist([R], [0], None, [256], [0, 256])  # 统计 R 通道 0 到 255 各灰度级对应的像素数
hist_G = cv2.calcHist([G], [0], None, [256], [0, 256])  # 统计 G 通道 0 到 255 各灰度级对应的像素数
hist_B = cv2.calcHist([B], [0], None, [256], [0, 256])  # 统计 B 通道 0 到 255 各灰度级对应的像素数
hist_gray = cv2.calcHist([imgG41_gray], [0], None, [256], [0, 256])  # 统计灰度图像 0 到 255 各灰度级对应的像素数
median_R = int(np.median(R))  # 计算 R 通道像素中值
median_G = int(np.median(G))  # 计算 G 通道像素中值
median_B = int(np.median(B))  # 计算 B 通道像素中值
median_gray = int(np.median(imgG41_gray))  # 计算灰度图像像素中值
count_R = int(hist_R[median_R][0])  # 获取 R 通道中值灰度级对应的像素数
count_G = int(hist_G[median_G][0])  # 获取 G 通道中值灰度级对应的像素数
count_B = int(hist_B[median_B][0])  # 获取 B 通道中值灰度级对应的像素数
count_gray = int(hist_gray[median_gray][0])  # 获取灰度图像中值灰度级对应的像素数
print(f"R通道中值为 {median_R}，对应像素数为 {count_R}")  # 输出 R 通道中值及其对应像素数
print(f"G通道中值为 {median_G}，对应像素数为 {count_G}")  # 输出 G 通道中值及其对应像素数
print(f"B通道中值为 {median_B}，对应像素数为 {count_B}")  # 输出 B 通道中值及其对应像素数
print(f"灰度图像中值为 {median_gray}，对应像素数为 {count_gray}")  # 输出灰度图像中值及其对应像素数
plt.figure("RGB Histogram")  # 创建 RGB 彩色图像直方图窗口
plt.plot(hist_R, color="red", label="R")  # 绘制 R 通道直方图
plt.plot(hist_G, color="green", label="G")  # 绘制 G 通道直方图
plt.plot(hist_B, color="blue", label="B")  # 绘制 B 通道直方图
plt.title("RGB Histogram")  # 设置 RGB 直方图标题
plt.xlabel("Pixel Value")  # 设置横坐标为像素灰度值
plt.ylabel("Pixel Count")  # 设置纵坐标为像素数量
plt.legend()  # 显示图例
plt.figure("Gray Histogram")  # 创建灰度图像直方图窗口
plt.plot(hist_gray, color="black", label="Gray")  # 绘制灰度图像直方图
plt.title("Gray Histogram")  # 设置灰度直方图标题
plt.xlabel("Pixel Value")  # 设置横坐标为像素灰度值
plt.ylabel("Pixel Count")  # 设置纵坐标为像素数量
plt.legend()  # 显示图例
plt.show()  # 显示所有 Matplotlib 绘图窗口
###################################################


