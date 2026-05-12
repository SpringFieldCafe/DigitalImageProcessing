import numpy as np  # 导入 NumPy 库，用于数组和数值计算
import cv2  # 导入 OpenCV 库，用于图像读取、处理和显示
from pathlib import Path  # 导入 Path 类，用于跨平台处理文件路径
from matplotlib import pyplot as plt  # 导入 Matplotlib 绘图模块，用于绘制直方图
import os  # 导入 os 模块，用于系统路径等相关操作

code_dir=Path(__file__).resolve().parent  # 获取当前代码文件所在目录
assignment2_dir=code_dir.resolve().parent  # 获取 assignment2 文件夹路径
src_dir=assignment2_dir.resolve()/"resource"  # 拼接 resource 资源文件夹路径
g41_path=src_dir/"g41_0.jpg"  # 拼接 g41_0.jpg 图像路径
miku_path=src_dir/"miku_0.jpg"  # 拼接 miku_0.jpg 图像路径

def stackImages(scale,imgArray):  # 定义图像拼接函数，scale 为缩放比例，imgArray 为待拼接图像数组
    rowsAvailable=isinstance(imgArray[0],list)  # 判断输入图像数组是否为二维列表
    if rowsAvailable:
        width=imgArray[0][0].shape[1]  # 获取基准图像宽度
        height=imgArray[0][0].shape[0]  # 获取基准图像高度
        rows=len(imgArray)  # 获取图像拼接的行数
        cols= max(len(row) for row in imgArray)  # 获取图像拼接的最大列数
        imgBlack = np.zeros_like(imgArray[0][0])  # 创建与基准图像大小相同的黑色占位图
        for x in range(0,rows):
            while len(imgArray[x]) < cols:  # 当当前行图像数量不足最大列数时继续补齐
                imgArray[x].append(imgBlack.copy())  # 用黑色图像补齐当前行列数
            for y in range(0,cols):  # 遍历当前行中的每一列图像
                if imgArray[x][y].shape[:2]==imgArray[0][0].shape[:2]:  # 判断当前图像尺寸是否与基准图像一致
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)  # 按比例缩放与基准尺寸一致的图像
                else:  # 当前图像尺寸与基准图像不一致时执行统一尺寸处理
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)  # 先统一到基准尺寸再按比例缩放
                if len(imgArray[x][y].shape)==2:
                    imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)  # 将灰度图转换为三通道 BGR 图像
        imgBlack=np.zeros((height,width,3),np.uint8)  # 创建三通道黑色图像
        hor=[imgBlack]*rows  # 初始化每一行横向拼接后的图像列表
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])  # 对当前行图像进行横向拼接
        ver=np.vstack(hor)  # 将各行结果纵向拼接成最终图像
    else:
        width=imgArray[0].shape[1]  # 获取一维图像列表中基准图像宽度
        height=imgArray[0].shape[0]  # 获取一维图像列表中基准图像高度
        cols=len(imgArray)  # 获取一维图像列表的图像数量

        for x in range(0,cols):  # 遍历一维图像列表中的每一张图像
            if imgArray[x].shape[:2]==imgArray[0].shape[:2]:  # 判断当前图像尺寸是否与基准图像一致
                imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)  # 按比例缩放与基准尺寸一致的图像
            else:  # 当前图像尺寸与基准图像不一致时执行统一尺寸处理
                imgArray[x]=cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)  # 先统一到基准尺寸再按比例缩放
            if len(imgArray[x].shape)==2:
                imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)  # 将灰度图转换为三通道 BGR 图像
        hor=np.hstack(imgArray)  # 将一维列表中的图像横向拼接
        ver=hor  # 一维拼接时最终结果即为横向拼接结果
    return ver  # 返回拼接后的图像


#######################################
#1
imgG41=cv2.imread(str(g41_path))  # 读取 g41_0.jpg 原始图像
imgG41_hsv = cv2.cvtColor(imgG41, cv2.COLOR_BGR2HSV)  # BGR 转 HSV 色彩空间
imgG41_hls = cv2.cvtColor(imgG41, cv2.COLOR_BGR2HLS)  # BGR 转 HLS 色彩空间
imgG41_lab = cv2.cvtColor(imgG41, cv2.COLOR_BGR2LAB)  # BGR 转 LAB 色彩空间
imgG41_yuv = cv2.cvtColor(imgG41, cv2.COLOR_BGR2YUV)  # BGR 转 YUV 色彩空间
imgG41stack=stackImages(0.3,[[imgG41_hls,imgG41_hsv],  # 调用图像拼接函数准备显示颜色空间结果
                             [imgG41_lab,imgG41_yuv]])  # 拼接不同颜色空间转换后的图像
cv2.imshow('stack',imgG41stack)  # 显示颜色空间转换结果拼接图
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
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


###########################################
#6
# 图像平滑技术主要用于去除噪声、减弱细节和降低图像灰度突变，常见方法有均值滤波、中值滤波、方框滤波、高斯滤波、双边滤波等
# 均值滤波属于线性滤波，通过邻域像素平均值替代中心像素，能够平滑图像，但容易造成边缘模糊
# 中值滤波属于非线性滤波，通过邻域像素中值替代中心像素，对椒盐噪声有较好抑制效果，并且比均值滤波更能保护边缘
# 方框滤波属于线性滤波，本质上使用矩形窗口对邻域像素求和或求平均，归一化方框滤波效果与均值滤波相近
imgG41_mean = cv2.blur(imgG41, (5, 5))  # 使用 5×5 滤波核对原图进行均值滤波处理
imgG41_median = cv2.medianBlur(imgG41, 5)  # 使用 5×5 滤波核对原图进行中值滤波处理，核大小必须为奇数
imgG41_box = cv2.boxFilter(imgG41, -1, (5, 5), normalize=True)  # 使用 5×5 滤波核对原图进行归一化方框滤波处理
imgG41_smooth_stack = stackImages(0.3, [[imgG41, imgG41_mean], [imgG41_median, imgG41_box]])  # 将原图、均值滤波图、中值滤波图、方框滤波图拼接显示
cv2.imshow("smooth_filter_compare", imgG41_smooth_stack)  # 显示不同平滑滤波方法的对比结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 处理效果分析：均值滤波可以整体平滑图像，但会使图像边缘和细节变模糊
# 处理效果分析：中值滤波对孤立噪声点的去除效果较好，同时对边缘的保护能力相对更强
# 处理效果分析：方框滤波在 normalize=True 时与均值滤波效果接近，能够降低噪声，但同样会带来一定模糊
##################################################



###########################################
#7
# 高斯滤波是一种常用的图像平滑方法，它通过高斯核对邻域像素加权平均，可以有效减弱图像噪声
# 高斯滤波中卷积核越大，参与计算的邻域范围越大，图像平滑效果越明显，但边缘和细节也会更加模糊
# 本实验分别选用 3×3、5×5、7×7、9×9、11×11、15×15 六种不同大小的高斯卷积核进行对比
imgG41_gauss_3 = cv2.GaussianBlur(imgG41, (3, 3), 0)  # 使用 3×3 高斯核对图像进行滤波，平滑程度较弱，细节保留较多
imgG41_gauss_5 = cv2.GaussianBlur(imgG41, (5, 5), 0)  # 使用 5×5 高斯核对图像进行滤波，噪声进一步减弱
imgG41_gauss_7 = cv2.GaussianBlur(imgG41, (7, 7), 0)  # 使用 7×7 高斯核对图像进行滤波，图像整体更加平滑
imgG41_gauss_9 = cv2.GaussianBlur(imgG41, (9, 9), 0)  # 使用 9×9 高斯核对图像进行滤波，边缘细节开始明显模糊
imgG41_gauss_11 = cv2.GaussianBlur(imgG41, (11, 11), 0)  # 使用 11×11 高斯核对图像进行滤波，平滑效果更强
imgG41_gauss_15 = cv2.GaussianBlur(imgG41, (15, 15), 0)  # 使用 15×15 高斯核对图像进行滤波，图像模糊程度最明显
imgG41_gauss_stack = stackImages(0.3, [[imgG41_gauss_3, imgG41_gauss_5, imgG41_gauss_7], [imgG41_gauss_9, imgG41_gauss_11, imgG41_gauss_15]])  # 按 2 行 3 列拼接六种高斯滤波结果
cv2.imshow("gaussian_filter_compare", imgG41_gauss_stack)  # 显示不同卷积核大小的高斯滤波对比结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 实验结果分析：3×3 和 5×5 高斯滤波后图像变化较小，能够轻微去噪并较好保留细节
# 实验结果分析：7×7 和 9×9 高斯滤波后图像平滑效果更明显，但部分边缘和纹理细节开始减弱
# 实验结果分析：11×11 和 15×15 高斯滤波后图像模糊程度较强，噪声减少更多，但图像细节损失也更明显
###################################################################


###########################################
#8
# 可实现图像锐化的滤波器主要有拉普拉斯算子、Sobel算子、Scharr算子、高通滤波器、非锐化掩蔽、自定义锐化卷积核等
# 拉普拉斯算子可以突出图像灰度变化剧烈的区域，常用于边缘增强和图像锐化
# 平滑卷积核通常通过邻域加权平均减弱噪声和细节，使图像变得更平滑
# 锐化卷积核通常增强中心像素并抑制周围像素，使边缘和纹理细节更加明显
kernel_mean_3 = np.ones((3, 3), np.float32) / 9  # 创建 3×3 均值平滑卷积核
kernel_mean_5 = np.ones((5, 5), np.float32) / 25  # 创建 5×5 均值平滑卷积核
kernel_gauss_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16  # 创建 3×3 类高斯平滑卷积核
kernel_sharp_3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # 创建 3×3 普通锐化卷积核
kernel_strong_sharp_3 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)  # 创建 3×3 强锐化卷积核
imgG41_mean_3 = cv2.filter2D(imgG41, -1, kernel_mean_3)  # 使用 3×3 均值卷积核对图像进行平滑处理
imgG41_mean_5 = cv2.filter2D(imgG41, -1, kernel_mean_5)  # 使用 5×5 均值卷积核对图像进行平滑处理
imgG41_gauss_3 = cv2.filter2D(imgG41, -1, kernel_gauss_3)  # 使用 3×3 类高斯卷积核对图像进行平滑处理
imgG41_lap = cv2.Laplacian(imgG41, cv2.CV_64F, ksize=3)  # 使用 3×3 拉普拉斯算子提取图像二阶边缘信息
imgG41_lap_sharp = np.clip(imgG41.astype(np.float64) - imgG41_lap, 0, 255).astype(np.uint8)  # 原图减去拉普拉斯响应，实现拉普拉斯锐化
imgG41_sharp_3 = cv2.filter2D(imgG41, -1, kernel_sharp_3)  # 使用 3×3 普通锐化卷积核增强图像细节
imgG41_strong_sharp_3 = cv2.filter2D(imgG41, -1, kernel_strong_sharp_3)  # 使用 3×3 强锐化卷积核进一步增强边缘细节
imgG41_mean_3_text = imgG41_mean_3.copy()  # 复制 3×3 均值滤波结果，避免文字直接影响原结果图
imgG41_mean_5_text = imgG41_mean_5.copy()  # 复制 5×5 均值滤波结果，避免文字直接影响原结果图
imgG41_gauss_3_text = imgG41_gauss_3.copy()  # 复制 3×3 类高斯滤波结果，避免文字直接影响原结果图
imgG41_lap_sharp_text = imgG41_lap_sharp.copy()  # 复制 3×3 拉普拉斯锐化结果，避免文字直接影响原结果图
imgG41_sharp_3_text = imgG41_sharp_3.copy()  # 复制 3×3 普通锐化结果，避免文字直接影响原结果图
imgG41_strong_sharp_3_text = imgG41_strong_sharp_3.copy()  # 复制 3×3 强锐化结果，避免文字直接影响原结果图
cv2.putText(imgG41_mean_3_text, "Mean 3x3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 3×3 均值滤波
cv2.putText(imgG41_mean_5_text, "Mean 5x5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 5×5 均值滤波
cv2.putText(imgG41_gauss_3_text, "Gauss 3x3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 3×3 类高斯滤波
cv2.putText(imgG41_lap_sharp_text, "Laplacian 3x3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 3×3 拉普拉斯锐化
cv2.putText(imgG41_sharp_3_text, "Sharp 3x3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 3×3 普通锐化
cv2.putText(imgG41_strong_sharp_3_text, "Strong Sharp 3x3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # 在图像上标注 3×3 强锐化
imgG41_filter_stack = stackImages(0.3, [[imgG41_mean_3_text, imgG41_mean_5_text, imgG41_gauss_3_text], [imgG41_lap_sharp_text, imgG41_sharp_3_text, imgG41_strong_sharp_3_text]])  # 按 2 行 3 列拼接六种滤波结果
cv2.imshow("smooth_sharpen_filter_compare", imgG41_filter_stack)  # 显示平滑和锐化滤波处理结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 实验结果分析：3×3 均值滤波平滑程度较弱，5×5 均值滤波平滑程度更强，但图像细节损失也更明显
# 实验结果分析：3×3 类高斯滤波会根据邻域权重进行平滑，相比普通均值滤波通常能更自然地减弱噪声
# 实验结果分析：拉普拉斯锐化和自定义锐化卷积核可以增强边缘与纹理，但强锐化卷积核可能使噪声和边缘过度增强
########################################################################3