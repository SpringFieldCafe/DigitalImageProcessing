import os
import numpy as np
from matplotlib import pyplot
from pathlib import Path
import cv2

main_path=os.path.abspath(__file__)
code_dir=os.path.dirname(main_path)
assignment3_dir=os.path.abspath(os.path.join(code_dir,".."))
resource_dir=Path(assignment3_dir)/"resource"
result_dir=Path(assignment3_dir)/"result"
suomi_path=resource_dir/"suomi.jpg"

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


###############################################
#1
imgSuomi = cv2.imread(str(suomi_path), cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取 suomi.jpg 图像
h, w = imgSuomi.shape  # 获取原始灰度图像的高度和宽度
pad_h = (8 - h % 8) % 8  # 计算高度方向需要补齐的像素数，使高度能被 8 整除
pad_w = (8 - w % 8) % 8  # 计算宽度方向需要补齐的像素数，使宽度能被 8 整除
imgPad = cv2.copyMakeBorder(imgSuomi, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)  # 对图像边缘进行复制填充，保证图像尺寸适合 8×8 分块
imgFloat = imgPad.astype(np.float32) - 128  # 将图像转换为 float32 类型并中心化，便于进行 DCT 变换
zigzag_index = [(i, j) for s in range(15) for i in range(8) for j in range(8) if i + j == s]  # 生成 8×8 块的 Zigzag 扫描顺序索引
zigzag_index = sorted(zigzag_index, key=lambda x: (x[0] + x[1], x[0] if (x[0] + x[1]) % 2 == 0 else -x[0]))  # 对 Zigzag 索引进行排序，使低频系数优先排列
keep_num = 32  # 每个 8×8 DCT 块共有 64 个系数，保留 50% 即保留 32 个系数
region_dct = np.zeros_like(imgFloat, dtype=np.float32)  # 创建区域编码后的 DCT 系数矩阵
threshold_dct = np.zeros_like(imgFloat, dtype=np.float32)  # 创建阈值编码后的 DCT 系数矩阵
for y in range(0, imgFloat.shape[0], 8):  # 按 8 个像素为步长遍历图像行方向
    for x in range(0, imgFloat.shape[1], 8):  # 按 8 个像素为步长遍历图像列方向
        block = imgFloat[y:y + 8, x:x + 8]  # 取出当前 8×8 图像块
        dct_block = cv2.dct(block)  # 对当前 8×8 图像块进行 DCT 变换
        region_block = np.zeros((8, 8), dtype=np.float32)  # 创建当前块的区域编码 DCT 系数块
        for i, j in zigzag_index[:keep_num]:  # 按 Zigzag 顺序保留前 32 个低频 DCT 系数
            region_block[i, j] = dct_block[i, j]  # 将低频区域内的 DCT 系数保留下来
        abs_block = np.abs(dct_block)  # 计算当前 DCT 系数块中每个系数的绝对值
        threshold_value = np.sort(abs_block.reshape(-1))[-keep_num]  # 找到绝对值排名前 32 的系数对应的阈值
        threshold_block = np.where(abs_block >= threshold_value, dct_block, 0)  # 保留绝对值最大的 32 个 DCT 系数，其余系数置 0
        region_dct[y:y + 8, x:x + 8] = region_block  # 保存当前块区域编码后的 DCT 系数
        threshold_dct[y:y + 8, x:x + 8] = threshold_block  # 保存当前块阈值编码后的 DCT 系数
region_decode = np.zeros_like(imgFloat, dtype=np.float32)  # 创建区域编码解码后的图像矩阵
threshold_decode = np.zeros_like(imgFloat, dtype=np.float32)  # 创建阈值编码解码后的图像矩阵
for y in range(0, imgFloat.shape[0], 8):  # 按 8 个像素为步长遍历区域编码 DCT 系数的行方向
    for x in range(0, imgFloat.shape[1], 8):  # 按 8 个像素为步长遍历区域编码 DCT 系数的列方向
        region_decode[y:y + 8, x:x + 8] = cv2.idct(region_dct[y:y + 8, x:x + 8])  # 对区域编码后的 8×8 DCT 系数块进行 IDCT 解码
        threshold_decode[y:y + 8, x:x + 8] = cv2.idct(threshold_dct[y:y + 8, x:x + 8])  # 对阈值编码后的 8×8 DCT 系数块进行 IDCT 解码
region_decode = np.clip(region_decode + 128, 0, 255).astype(np.uint8)  # 将区域编码解码图像反中心化并限制到 0 到 255
threshold_decode = np.clip(threshold_decode + 128, 0, 255).astype(np.uint8)  # 将阈值编码解码图像反中心化并限制到 0 到 255
region_decode = region_decode[:h, :w]  # 去除区域编码解码图像中之前补齐的边缘部分
threshold_decode = threshold_decode[:h, :w]  # 去除阈值编码解码图像中之前补齐的边缘部分
mse_region = np.mean((imgSuomi.astype(np.float32) - region_decode.astype(np.float32)) ** 2)  # 计算原图与区域编码解码图像之间的均方误差
mse_threshold = np.mean((imgSuomi.astype(np.float32) - threshold_decode.astype(np.float32)) ** 2)  # 计算原图与阈值编码解码图像之间的均方误差
psnr_region = 10 * np.log10(255 * 255 / mse_region) if mse_region != 0 else float("inf")  # 计算区域编码解码图像的 PSNR 值
psnr_threshold = 10 * np.log10(255 * 255 / mse_threshold) if mse_threshold != 0 else float("inf")  # 计算阈值编码解码图像的 PSNR 值
print(f"区域编码保留 50% 系数后的 MSE：{mse_region:.2f}，PSNR：{psnr_region:.2f} dB")  # 输出区域编码解码图像的误差和峰值信噪比
print(f"阈值编码保留 50% 大系数后的 MSE：{mse_threshold:.2f}，PSNR：{psnr_threshold:.2f} dB")  # 输出阈值编码解码图像的误差和峰值信噪比
imgCompare = stackImages(0.4, [[imgSuomi, region_decode, threshold_decode]])  # 将原图、区域编码解码图、阈值编码解码图横向拼接比较
cv2.imshow("DCT_compression_compare", imgCompare)  # 显示 DCT 图像压缩解码结果对比图
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 实验结果分析：区域编码按照固定低频区域保留系数，能够保留图像主要轮廓，但部分细节可能丢失
# 实验结果分析：阈值编码按照 DCT 系数大小保留重要系数，通常能保留更多能量较大的信息，重建图像质量可能更好
# 实验结果分析：两种方法都只保留 50% 的 DCT 系数，因此都能实现一定压缩，但解码图像会出现不同程度的细节损失
###############################################

###############################################
#2
# 高帽运算是原图减去开运算结果，主要用于提取图像中较小的亮区域或细节
# 开运算是先腐蚀再膨胀，可以去除较小的白色噪声点，同时保持较大目标区域
result_dir.mkdir(parents=True, exist_ok=True)  # 如果 result 文件夹不存在，则自动创建
j_path = result_dir / "j.png"  # 设置生成的 j.png 图像保存路径
imgJ = np.zeros((240, 240), dtype=np.uint8)  # 使用 NumPy 创建一张 240×240 的黑色单通道图像
imgJ[30:50, 60:180] = 255  # 使用数组切片绘制字母 J 的上方横线
imgJ[50:170, 130:150] = 255  # 使用数组切片绘制字母 J 的右侧竖线
imgJ[150:170, 70:150] = 255  # 使用数组切片绘制字母 J 的下方横线
imgJ[130:170, 60:80] = 255  # 使用数组切片绘制字母 J 的左侧弯钩部分
imgJ[80:83, 40:43] = 255  # 添加一个较小的白色亮点，用于观察高帽运算效果
imgJ[110:113, 200:203] = 255  # 添加一个较小的白色亮点，用于观察高帽运算效果
imgJ[190:193, 120:123] = 255  # 添加一个较小的白色亮点，用于观察高帽运算效果
cv2.imwrite(str(j_path), imgJ)  # 将 NumPy 生成的二值图像保存为 result 文件夹下的 j.png
imgJ_read = cv2.imread(str(j_path), cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取刚刚生成的 j.png 图像
if imgJ_read is None:  # 判断 j.png 是否读取成功
    raise FileNotFoundError(f"图片读取失败：{j_path}")  # 如果读取失败，则抛出文件路径错误
_, imgJ_bin = cv2.threshold(imgJ_read, 127, 255, cv2.THRESH_BINARY)  # 对图像进行二值化处理，确保图像只有黑白两类像素
kernel_5x5 = np.ones((5, 5), dtype=np.uint8)  # 使用 NumPy 生成 5×5 的正方形结构元素
imgJ_erode = cv2.erode(imgJ_bin, kernel_5x5, iterations=1)  # 使用 cv2.erode() 对二值图像进行腐蚀运算
imgJ_dilate = cv2.dilate(imgJ_bin, kernel_5x5, iterations=1)  # 使用 cv2.dilate() 对二值图像进行膨胀运算
imgJ_open = cv2.dilate(cv2.erode(imgJ_bin, kernel_5x5, iterations=1), kernel_5x5, iterations=1)  # 先腐蚀再膨胀，实现开运算
imgJ_tophat = cv2.subtract(imgJ_bin, imgJ_open)  # 用原二值图像减去开运算图像，实现高帽运算
cv2.imwrite(str(result_dir / "j_erode.png"), imgJ_erode)  # 将腐蚀结果保存到 result 文件夹
cv2.imwrite(str(result_dir / "j_dilate.png"), imgJ_dilate)  # 将膨胀结果保存到 result 文件夹
cv2.imwrite(str(result_dir / "j_open.png"), imgJ_open)  # 将开运算结果保存到 result 文件夹
cv2.imwrite(str(result_dir / "j_tophat.png"), imgJ_tophat)  # 将高帽运算结果保存到 result 文件夹
imgJ_stack = stackImages(0.7, [[imgJ_bin, imgJ_erode, imgJ_dilate], [imgJ_open, imgJ_tophat, imgJ_read]])  # 将二值图、腐蚀图、膨胀图、开运算图、高帽图和原读取图按 2 行 3 列拼接
cv2.imshow("j_morphology_compare", imgJ_stack)  # 显示 j.png 的形态学处理结果对比图
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 实验结果分析：腐蚀运算会使白色目标区域变小，细小白点可能被去除
# 实验结果分析：膨胀运算会使白色目标区域变大，字母 J 的笔画会变粗
# 实验结果分析：高帽运算可以提取原图中被开运算去除的小亮点或细小亮区域
##################################################

###############################################
#3
# Azure 颜色在 RGB 中为 (240,255,255)，OpenCV 使用 BGR 顺序，所以这里写成 (255,255,240)
imgSuomi = cv2.imread(str(suomi_path), cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取 suomi.jpg 图像
if imgSuomi is None:  # 判断图像是否读取成功
    raise FileNotFoundError(f"图片读取失败：{suomi_path}")  # 如果读取失败，抛出文件路径错误
azure_color = (255, 255, 240)  # 设置文字颜色为 Azure，注意 OpenCV 中颜色顺序为 BGR
kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)  # 定义 Roberts 算子 X 方向卷积核
kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # 定义 Roberts 算子 Y 方向卷积核
roberts_x = cv2.filter2D(imgSuomi, cv2.CV_32F, kernel_roberts_x)  # 使用 Roberts X 方向算子进行边缘检测
roberts_y = cv2.filter2D(imgSuomi, cv2.CV_32F, kernel_roberts_y)  # 使用 Roberts Y 方向算子进行边缘检测
img_roberts = cv2.convertScaleAbs(np.sqrt(roberts_x ** 2 + roberts_y ** 2))  # 合成 Roberts 两个方向的边缘检测结果
kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)  # 定义 Prewitt 算子 X 方向卷积核
kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)  # 定义 Prewitt 算子 Y 方向卷积核
prewitt_x = cv2.filter2D(imgSuomi, cv2.CV_32F, kernel_prewitt_x)  # 使用 Prewitt X 方向算子进行边缘检测
prewitt_y = cv2.filter2D(imgSuomi, cv2.CV_32F, kernel_prewitt_y)  # 使用 Prewitt Y 方向算子进行边缘检测
img_prewitt = cv2.convertScaleAbs(np.sqrt(prewitt_x ** 2 + prewitt_y ** 2))  # 合成 Prewitt 两个方向的边缘检测结果
sobel_x = cv2.Sobel(imgSuomi, cv2.CV_32F, 1, 0, ksize=3)  # 使用 Sobel 算子计算 X 方向梯度
sobel_y = cv2.Sobel(imgSuomi, cv2.CV_32F, 0, 1, ksize=3)  # 使用 Sobel 算子计算 Y 方向梯度
img_sobel = cv2.convertScaleAbs(np.sqrt(sobel_x ** 2 + sobel_y ** 2))  # 合成 Sobel 两个方向的边缘检测结果
laplacian = cv2.Laplacian(imgSuomi, cv2.CV_32F, ksize=3)  # 使用拉普拉斯算子进行二阶边缘检测
img_laplacian = cv2.convertScaleAbs(laplacian)  # 将拉普拉斯边缘检测结果转换为 uint8 图像
img_gaussian = cv2.GaussianBlur(imgSuomi, (5, 5), 0)  # 先对图像进行高斯平滑，减少噪声对 LoG 算子的影响
log_edge = cv2.Laplacian(img_gaussian, cv2.CV_32F, ksize=3)  # 对高斯平滑后的图像使用拉普拉斯算子，实现 LoG 边缘检测
img_log = cv2.convertScaleAbs(log_edge)  # 将 LoG 边缘检测结果转换为 uint8 图像
img_canny = cv2.Canny(imgSuomi, 80, 160)  # 使用 Canny 算子进行边缘检测，80 和 160 分别为低阈值和高阈值
img_roberts_text = cv2.cvtColor(img_roberts, cv2.COLOR_GRAY2BGR)  # 将 Roberts 边缘图转换为三通道图像，便于添加彩色文字
img_prewitt_text = cv2.cvtColor(img_prewitt, cv2.COLOR_GRAY2BGR)  # 将 Prewitt 边缘图转换为三通道图像，便于添加彩色文字
img_sobel_text = cv2.cvtColor(img_sobel, cv2.COLOR_GRAY2BGR)  # 将 Sobel 边缘图转换为三通道图像，便于添加彩色文字
img_laplacian_text = cv2.cvtColor(img_laplacian, cv2.COLOR_GRAY2BGR)  # 将拉普拉斯边缘图转换为三通道图像，便于添加彩色文字
img_log_text = cv2.cvtColor(img_log, cv2.COLOR_GRAY2BGR)  # 将 LoG 边缘图转换为三通道图像，便于添加彩色文字
img_canny_text = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)  # 将 Canny 边缘图转换为三通道图像，便于添加彩色文字
cv2.putText(img_roberts_text, "Roberts", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在 Roberts 边缘图上添加 Azure 颜色文字
cv2.putText(img_prewitt_text, "Prewitt", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在 Prewitt 边缘图上添加 Azure 颜色文字
cv2.putText(img_sobel_text, "Sobel", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在 Sobel 边缘图上添加 Azure 颜色文字
cv2.putText(img_laplacian_text, "Laplacian", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在拉普拉斯边缘图上添加 Azure 颜色文字
cv2.putText(img_log_text, "LoG", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在 LoG 边缘图上添加 Azure 颜色文字
cv2.putText(img_canny_text, "Canny", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azure_color, 3)  # 在 Canny 边缘图上添加 Azure 颜色文字
imgEdgeStack = stackImages(0.5, [[img_roberts_text, img_prewitt_text, img_sobel_text], [img_laplacian_text, img_log_text, img_canny_text]])  # 将六种边缘检测结果按 2 行 3 列拼接
cv2.imshow("edge_detection_compare", imgEdgeStack)  # 显示六种边缘检测算子的对比结果
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
# 实验结果分析：Roberts 算子卷积核较小，对细节变化敏感，但抗噪声能力较弱
# 实验结果分析：Prewitt 算子能够检测水平和垂直方向边缘，效果比 Roberts 更稳定
# 实验结果分析：Sobel 算子在计算梯度时具有一定平滑作用，边缘检测效果通常比 Prewitt 更清晰
# 实验结果分析：拉普拉斯算子属于二阶微分算子，可以突出灰度突变区域，但也容易放大噪声
# 实验结果分析：LoG 算子先高斯平滑再拉普拉斯检测，能减弱噪声影响，边缘结果比单独拉普拉斯更平滑
# 实验结果分析：Canny 算子包含平滑、梯度计算、非极大值抑制和双阈值连接，边缘通常更细、更连续
##################################################################################################