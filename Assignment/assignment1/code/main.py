import cv2 # 导入OpenCV库，用于图像读取、显示、保存和处理
import numpy as np # 导入NumPy库，用于数组和矩阵运算
from pathlib import Path # 导入Path类，用于跨平台拼接文件路径
import os # 导入os模块，用于获取文件路径、拼接路径和处理系统路径
from matplotlib import pyplot as plt # 导入matplotlib中的pyplot模块，用于图像可视化显示
import sys # 导入sys模块，用于获取Python解释器信息和输出运行提示

path_main = Path(os.path.abspath(__file__)).resolve().parents[1] # 使用os获取当前文件绝对路径，再转换为Path对象，并获取assignment1文件夹路径
resource_dir = path_main / "resource" # 使用Path拼接resource资源文件夹路径
result_dir = Path(os.path.join(str(path_main), "result")) # 使用os.path.join拼接result结果文件夹路径，并转换为Path对象
an94_path = Path(resource_dir / 'an94.jpg') # 使用Path拼接an94.jpg图像文件路径
np1_path = Path(resource_dir / 'np1.jpg') # 使用Path拼接np1.jpg图像文件路径
pics_dir = Path(os.path.join(str(result_dir), "pics")) # 使用os.path.join拼接pics文件夹路径，并转换为Path对象
ppics_dir = pics_dir / "ppics" # 使用Path拼接pics文件夹下ppics子目录路径
SHE_path = Path(resource_dir / 'SHE.jpg') # 使用Path拼接SHE.jpg图像文件路径

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


##########################
# 1
img = cv2.imread(str(resource_dir / "PostgreSQL.jpg"))  # 使用cv2.imread()函数读取PostgreSQL图片
print(img)  # 输出读取到的图像像素值矩阵
##########################


################################
# 2
cv2.imshow("img", img)  # 使用cv2.imshow()函数显示读取到的图像
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
###################################

#####################################
# 3
save_dir = pics_dir  # 设置图像保存目录为assignment1/result/pics文件夹
save_dir.mkdir(parents=True, exist_ok=True)  # 如果result/pics文件夹不存在，则自动创建
save_path = save_dir / "Postgre.jpg"  # 设置保存后的图像文件路径和文件名
cv2.imwrite(str(save_path), img)  # 使用cv2.imwrite()函数将图像保存到result/pics文件夹下
########################################

#####################################
# 4
src_dir = pics_dir  # 设置读取图片的源文件夹为assignment1/result/pics
dst_dir = ppics_dir  # 设置保存图片的目标文件夹为assignment1/result/pics/ppics
dst_dir.mkdir(parents=True, exist_ok=True)  # 如果ppics子目录不存在，则自动创建
img_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  # 定义允许读取的图片文件后缀
for img_path in src_dir.iterdir():  # 遍历result/pics文件夹下的所有文件和文件夹
    if not img_path.is_file():  # 判断当前路径是否不是文件
        continue  # 如果不是文件，则跳过本次循环
    if img_path.suffix.lower() not in img_suffix:  # 判断当前文件是否不是图片格式
        continue  # 如果不是图片文件，则跳过本次循环
    img_temp = cv2.imread(str(img_path))  # 使用cv2.imread()函数读取当前图片
    if img_temp is None:  # 判断当前图片是否读取失败
        print(f"图片读取失败：{img_path}")  # 输出读取失败的图片路径
        continue  # 跳过读取失败的图片
    cv2.imshow("image", img_temp)  # 使用cv2.imshow()函数显示当前读取到的图片
    save_path = dst_dir / img_path.name  # 设置当前图片保存到ppics子目录中的路径
    cv2.imwrite(str(save_path), img_temp)  # 使用cv2.imwrite()函数将当前图片保存到ppics子目录中
    cv2.waitKey(0)  # 等待键盘按键后继续处理下一张图片
cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
########################################

######################################################
# 5
h, w = img.shape[:2]  # 获取图像的高度和宽度
channels = img.shape[2]  # 获取图像的通道数
pixel_count = h * w  # 计算图像的像素数
dtype = img.dtype  # 获取图像的数据类型
print("图像高度：", h)  # 输出图像高度
print("图像宽度：", w)  # 输出图像宽度
print("图像像素数：", pixel_count)  # 输出图像像素总数
print("图像通道数：", channels)  # 输出图像通道数
print("图像数据类型：", dtype)  # 输出图像数据类型
#########################################################


#######################################################
# 6
imgScale = cv2.resize(img, (200, 200))  # 将原图缩放到200×200大小
imgCv2Add = cv2.add(imgScale, imgScale)  # 使用OpenCV加法对图像进行相加
imgNpAdd = np.array(imgScale + imgScale)  # 使用NumPy数组加法对图像进行相加
imgCv2diff = cv2.subtract(imgCv2Add, imgScale)  # 使用OpenCV减法计算图像差值
imgNpdiff = np.array(imgScale - imgNpAdd)  # 使用NumPy数组减法计算图像差值
imgGray = cv2.cvtColor(imgScale, cv2.COLOR_BGR2GRAY)  # 将缩放后的图像转换为灰度图
imgCv2mp = cv2.multiply(imgScale, imgScale)  # 使用OpenCV乘法对图像进行逐像素相乘
imgNpdot = np.dot(imgGray, imgGray)  # 使用NumPy的dot()函数进行矩阵乘法
den = (imgScale * 0.01).astype(np.uint8)  # 构造除法运算中的除数图像
den[den == 0] = 1  # 将除数中的0改为1，避免除以0
imgdivide = cv2.divide(imgScale, den)  # 使用cv2.divide()函数实现图像除法运算
cv2.imshow("c+", imgCv2Add)  # 显示OpenCV加法得到的图像
cv2.imshow("n+", imgNpAdd)  # 显示NumPy加法得到的图像
cv2.imshow("c-", imgCv2diff)  # 显示OpenCV减法结果
cv2.imshow("n-", imgNpdiff)  # 显示NumPy减法结果
cv2.imshow("c*", imgCv2mp)  # 显示OpenCV乘法结果
cv2.imshow("dot", imgNpdot)  # 显示NumPy矩阵乘法结果
cv2.imshow("div", imgdivide)  # 显示图像除法结果
cv2.waitKey()  # 等待键盘按键后继续执行程序
###############################################

###################################################
# 7
imgNai = cv2.imread(str(resource_dir / "nailong.png"))  # 读取resource文件夹中的自拍图像
img2 = cv2.resize(imgNai, (200, 200))  # 将自拍图像缩放到200×200大小
res = cv2.addWeighted(imgScale, 0.7, img2, 0.3, gamma=3)  # 按0.7和0.3权重融合两张图像，gamma设置为3增强显示效果
cv2.imshow("res", res)  # 显示图像融合结果
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
#####################################################

#########################################################
# 8
img8 = cv2.resize(img, (200, 200))  # 将原图缩放为200×200，方便显示和处理
h, w = img8.shape[:2]  # 获取图像的高度和宽度
mask = np.zeros((h, w), dtype=np.uint8)  # 构造一个与原图大小相同的全黑单通道掩膜
cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)  # 在掩膜中画一个白色矩形区域表示掩膜区域
img_and = cv2.bitwise_and(img8, img8, mask=mask)  # 使用按位与运算保留掩膜白色区域内的图像
mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 将单通道掩膜转换为三通道掩膜
img_or = cv2.bitwise_or(img8, mask3)  # 使用按位或运算去掉掩膜内的图像效果
img_xor = cv2.bitwise_xor(img8, mask3)  # 使用按位异或运算处理图像
# 用黑色描边 + 黄色文字
def add_label(image, text):  # 定义文字标记函数，用于给每张结果图像添加说明文字
    cv2.putText(image, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)  # 先绘制黑色粗文字作为描边，增强文字对比度
    cv2.putText(image, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)  # 再绘制黄色文字，使文字在不同背景上更清晰
    return image  # 返回已经添加文字标记的图像
img8 = add_label(img8, "Original")  # 给原图添加Original标记
mask = add_label(mask, "Mask")  # 给掩膜图添加Mask标记
img_and = add_label(img_and, "AND: Keep Mask Area")  # 给按位与结果图添加保留掩膜区域的标记
img_or = add_label(img_or, "OR: Remove Mask Area")  # 给按位或结果图添加去掉掩膜区域的标记
img_xor = add_label(img_xor, "XOR")  # 给按位异或结果图添加XOR标记
imgRes = stackImages(0.7, [[img8, mask, img_and], [img_or, img_xor]])  # 将原图、掩膜图和三种按位运算结果图拼接成一张展示图
cv2.imshow('s', imgRes)  # 显示拼接后的结果图像
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
#################################################################################


#######################################################
# 9
jpg_path = pics_dir / "Postgre.jpg"  # 设置result/pics文件夹下某一张JPG图像路径
img_jpg = cv2.imread(str(jpg_path))  # 读取result/pics文件夹下的JPG图像
img_not = cv2.bitwise_not(img_jpg)  # 对JPG图像进行按位非运算
cv2.imshow("bit", img_not)  # 显示按位非运算后的图像
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
################################################################

#######################################################################
# 10
cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
an94_path = Path(resource_dir / "an94.jpg")  # 使用Path拼接an94.jpg图像文件路径
img10 = cv2.imread(str(an94_path))  # 使用cv2.imread()读取an94.jpg图像
img10 = cv2.resize(img10, (350, 350))  # 将an94.jpg图像缩放为350×350大小
M_move = np.array([[1, 0, 60], [0, 1, 40]], dtype=np.float32)  # 设置仿射变换矩阵，实现向右60像素、向下40像素平移
dst_same = cv2.warpAffine(img10, M_move, (350, 350))  # 使用cv2.warpAffine()实现显示窗口大小不变的平移
dst_large = cv2.warpAffine(img10, M_move, (450, 550))  # 使用cv2.warpAffine()实现显示窗口大小改变的平移，宽度增加100，高度增加200
img10_text = img10.copy()  # 复制原图，避免直接修改原始图像
dst_same_text = dst_same.copy()  # 复制窗口大小不变的平移结果图
dst_large_text = dst_large.copy()  # 复制窗口大小改变的平移结果图
cv2.putText(img10_text, "Original 350x350", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (238, 130, 238), 2)  # 在原图上添加文字标注
cv2.putText(dst_same_text, "Same Window 350x350", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (238, 130, 238), 2)  # 在窗口大小不变结果图上添加文字标注
cv2.putText(dst_large_text, "Large Window 450x550", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (238, 130, 238), 2)  # 在窗口大小改变结果图上添加文字标注
img10_rgb = cv2.cvtColor(img10_text, cv2.COLOR_BGR2RGB)  # 将原图从BGR格式转换为RGB格式，便于matplotlib正确显示颜色
dst_same_rgb = cv2.cvtColor(dst_same_text, cv2.COLOR_BGR2RGB)  # 将窗口大小不变结果图从BGR格式转换为RGB格式
dst_large_rgb = cv2.cvtColor(dst_large_text, cv2.COLOR_BGR2RGB)  # 将窗口大小改变结果图从BGR格式转换为RGB格式
plt.figure(figsize=(12, 6))  # 创建画布并设置整体显示大小
plt.subplot(1, 3, 1)  # 创建1行3列中的第1个子图
plt.imshow(img10_rgb)  # 显示标注后的原图
plt.title("Original")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.subplot(1, 3, 2)  # 创建1行3列中的第2个子图
plt.imshow(dst_same_rgb)  # 显示窗口大小不变的平移结果
plt.title("Same Size Translation")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.subplot(1, 3, 3)  # 创建1行3列中的第3个子图
plt.imshow(dst_large_rgb)  # 显示窗口大小改变的平移结果
plt.title("Large Size Translation")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 使用matplotlib统一显示三张图像
############################################################################

#######################################################################
# 11
np1_path = Path(resource_dir / 'np1.jpg')  # 使用Path拼接np1.jpg图像文件路径
img11 = cv2.imread(str(np1_path))  # 使用cv2.imread()读取np1.jpg图像
img11 = cv2.resize(img11, (320, 320))  # 将np1图像缩放为320×320大小
h, w = img11.shape[:2]  # 获取图像的高度和宽度
scale_x = 0.75  # 设置水平方向缩放比例为0.75
scale_y = 0.75  # 设置垂直方向缩放比例为0.75
M_scale_ratio = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])  # 构造按比例缩放的仿射变换矩阵
img_scale_ratio = cv2.warpAffine(img11, M_scale_ratio, (int(w * scale_x), int(h * scale_y)))  # 使用warpAffine按缩放比例缩放图像
target_w = 275  # 设置目标图像宽度
target_h = 375  # 设置目标图像高度
scale_x2 = target_w / w  # 根据目标宽度计算水平方向缩放比例
scale_y2 = target_h / h  # 根据目标高度计算垂直方向缩放比例
M_scale_size = np.float32([[scale_x2, 0, 0], [0, scale_y2, 0]])  # 构造按指定大小缩放的仿射变换矩阵
img_scale_size = cv2.warpAffine(img11, M_scale_size, (target_w, target_h))  # 使用warpAffine将图像缩放到指定大小
center = (w // 2, h // 2)  # 设置旋转中心为图像中心
M_clockwise = cv2.getRotationMatrix2D(center, -45, 1.0)  # 构造顺时针旋转45度的旋转矩阵
img_clockwise = cv2.warpAffine(img11, M_clockwise, (w, h))  # 使用warpAffine实现顺时针旋转
M_counterclockwise = cv2.getRotationMatrix2D(center, 45, 1.0)  # 构造逆时针旋转45度的旋转矩阵
img_counterclockwise = cv2.warpAffine(img11, M_counterclockwise, (w, h))  # 使用warpAffine实现逆时针旋转
img11_text = img11.copy()  # 复制原图，避免直接修改原始图像
img_scale_ratio_text = img_scale_ratio.copy()  # 复制按比例缩放后的图像
img_scale_size_text = img_scale_size.copy()  # 复制按指定大小缩放后的图像
img_clockwise_text = img_clockwise.copy()  # 复制顺时针旋转后的图像
img_counterclockwise_text = img_counterclockwise.copy()  # 复制逆时针旋转后的图像
cv2.putText(img11_text, "Original 320x320", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)  # 在原图上添加文字标注
cv2.putText(img_scale_ratio_text, "Scale 0.75", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 130, 238), 1)  # 在按比例缩放图像上添加文字标注
cv2.putText(img_scale_size_text, "Size 275x375", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)  # 在按指定大小缩放图像上添加文字标注
cv2.putText(img_clockwise_text, "Clockwise 45", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)  # 在顺时针旋转图像上添加文字标注
cv2.putText(img_counterclockwise_text, "Counter 45", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)  # 在逆时针旋转图像上添加文字标注
cv2.imshow("n", img_scale_ratio_text)  # 使用OpenCV单独显示按比例缩放后的图像，窗口标题只用n方便拖动
cv2.waitKey(1)  # 等待1毫秒刷新OpenCV窗口，使Scale Ratio图像能够显示出来
img11_rgb = cv2.cvtColor(img11_text, cv2.COLOR_BGR2RGB)  # 将原图从BGR格式转换为RGB格式
img_scale_size_rgb = cv2.cvtColor(img_scale_size_text, cv2.COLOR_BGR2RGB)  # 将按指定大小缩放图像从BGR格式转换为RGB格式
img_clockwise_rgb = cv2.cvtColor(img_clockwise_text, cv2.COLOR_BGR2RGB)  # 将顺时针旋转图像从BGR格式转换为RGB格式
img_counterclockwise_rgb = cv2.cvtColor(img_counterclockwise_text, cv2.COLOR_BGR2RGB)  # 将逆时针旋转图像从BGR格式转换为RGB格式
plt.figure(figsize=(12, 5))  # 创建画布并设置整体显示大小
plt.subplot(1, 4, 1)  # 创建1行4列中的第1个子图
plt.imshow(img11_rgb)  # 显示带文字标注的原图
plt.title("Original")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.subplot(1, 4, 2)  # 创建1行4列中的第2个子图
plt.imshow(img_scale_size_rgb)  # 显示带文字标注的指定大小缩放图像
plt.title("Scale Size")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.subplot(1, 4, 3)  # 创建1行4列中的第3个子图
plt.imshow(img_clockwise_rgb)  # 显示带文字标注的顺时针旋转图像
plt.title("Clockwise")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.subplot(1, 4, 4)  # 创建1行4列中的第4个子图
plt.imshow(img_counterclockwise_rgb)  # 显示带文字标注的逆时针旋转图像
plt.title("Counterclockwise")  # 设置子图标题
plt.axis("off")  # 关闭坐标轴显示
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 使用matplotlib统一展示除Scale Ratio以外的其他图像
cv2.waitKey(0)  # 等待键盘按键后关闭OpenCV窗口
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
#####################################################################################################

########################################################################
# 12
img_path = pics_dir / "self.jpg"  # 设置自拍照片路径为assignment1/result/pics/self.jpg
if not img_path.exists():  # 判断自拍照片是否还没有放到pics文件夹中
    cv2.imwrite(str(img_path), imgNai)  # 如果没有自拍照片，则使用已读取的自拍图像生成self.jpg保证程序可执行
img12 = cv2.imread(str(img_path))  # 读取自拍图像
if img12 is None:  # 判断图像是否读取失败
    sys.stdout.write(f"自拍图像读取失败，请检查路径：{img_path}\n")  # 输出错误提示和自拍图像路径
else:  # 图像读取成功后执行
    sys.stdout.write("图像属性：\n")  # 输出图像属性标题
    sys.stdout.write(f"图像尺寸：{img12.shape}\n")  # 输出图像尺寸，包括高度、宽度和通道数
    sys.stdout.write(f"图像高度：{img12.shape[0]}\n")  # 输出图像高度
    sys.stdout.write(f"图像宽度：{img12.shape[1]}\n")  # 输出图像宽度
    sys.stdout.write(f"图像通道数：{img12.shape[2]}\n")  # 输出图像通道数
    sys.stdout.write(f"图像数据类型：{img12.dtype}\n")  # 输出图像数据类型
    h, w = img12.shape[:2]  # 获取原图高度和宽度
    img_crop = img12[:, :w // 2]  # 将原图像规则剪裁一半，这里保留左半部分
    save_dir = ppics_dir  # 设置剪裁后图像保存目录为assignment1/result/pics/ppics
    save_dir.mkdir(parents=True, exist_ok=True)  # 如果ppics子目录不存在，则创建
    save_path = save_dir / "p_self.jpg"  # 设置剪裁后图像保存路径
    cv2.imwrite(str(save_path), img_crop)  # 将剪裁后的图像保存到ppics子目录中
    img_crop_read = cv2.imread(str(save_path))  # 重新读取保存后的剪裁图像
    sys.stdout.write("图像属性：\n")  # 输出图像属性标题
    sys.stdout.write(f"图像尺寸：{img_crop_read.shape}\n")  # 输出图像尺寸
    sys.stdout.write(f"图像高度：{img_crop_read.shape[0]}\n")  # 输出图像高度
    sys.stdout.write(f"图像宽度：{img_crop_read.shape[1]}\n")  # 输出图像宽度
    sys.stdout.write(f"图像通道数：{img_crop_read.shape[2]}\n")  # 输出图像通道数
    sys.stdout.write(f"图像数据类型：{img_crop_read.dtype}\n")  # 输出图像数据类型
    cv2.imshow("p_38", img_crop_read)  # 在名为p_38的窗口中显示剪裁后的图像
    cv2.waitKey()  # 等待键盘按键后继续执行程序
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
############################################################################################

###################################################
# 13
SHE_path = Path(resource_dir / 'SHE.jpg')  # 使用Path拼接SHE.jpg图像文件路径
img13 = cv2.imread(str(SHE_path))  # 读取SHE.jpg图像
img13 = cv2.resize(img13, (420, 420))  # 将原图先缩放为420×420大小
img_flip_horizontal = cv2.flip(img13, 1)  # 使用cv2.flip函数实现水平镜像变换
img_flip_vertical = cv2.flip(img13, 0)  # 使用cv2.flip函数实现垂直镜像变换
img_flip_diagonal = cv2.flip(img13, -1)  # 使用cv2.flip函数实现对角镜像变换
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置OpenCV绘制文字使用的字体
color = (226, 43, 138)  # 设置文字颜色为接近blueviolet的BGR颜色
img_original_label = img13.copy()  # 复制原图用于添加文字标签
img_horizontal_label = img_flip_horizontal.copy()  # 复制水平镜像图用于添加文字标签
img_vertical_label = img_flip_vertical.copy()  # 复制垂直镜像图用于添加文字标签
img_diagonal_label = img_flip_diagonal.copy()  # 复制对角镜像图用于添加文字标签
cv2.putText(img_original_label, "Original", (18, 36), font, 0.9, color, 2)  # 给原图添加标签
cv2.putText(img_horizontal_label, "Horizontal Flip", (18, 36), font, 0.9, color, 2)  # 给水平镜像图添加标签
cv2.putText(img_vertical_label, "Vertical Flip", (18, 36), font, 0.9, color, 2)  # 给垂直镜像图添加标签
cv2.putText(img_diagonal_label, "Diagonal Flip", (18, 36), font, 0.9, color, 2)  # 给对角镜像图添加标签
img_original_rgb = cv2.cvtColor(img_original_label, cv2.COLOR_BGR2RGB)  # 将原图由BGR转换为RGB，便于plt显示
img_horizontal_rgb = cv2.cvtColor(img_horizontal_label, cv2.COLOR_BGR2RGB)  # 将水平镜像图由BGR转换为RGB
img_vertical_rgb = cv2.cvtColor(img_vertical_label, cv2.COLOR_BGR2RGB)  # 将垂直镜像图由BGR转换为RGB
img_diagonal_rgb = cv2.cvtColor(img_diagonal_label, cv2.COLOR_BGR2RGB)  # 将对角镜像图由BGR转换为RGB
fig = plt.figure(figsize=(9, 9))  # 创建plt画布并设置显示大小
gs = plt.GridSpec(3, 3, width_ratios=[1, 0.04, 1], height_ratios=[1, 0.04, 1])  # 创建3×3布局，中间行列作为很窄的空白间隔
ax1 = fig.add_subplot(gs[0, 0])  # 在左上角位置放置原图
ax2 = fig.add_subplot(gs[0, 2])  # 在右上角位置放置水平镜像图
ax3 = fig.add_subplot(gs[2, 0])  # 在左下角位置放置垂直镜像图
ax4 = fig.add_subplot(gs[2, 2])  # 在右下角位置放置对角镜像图
ax1.imshow(img_original_rgb)  # 显示原图
ax2.imshow(img_horizontal_rgb)  # 显示水平镜像图
ax3.imshow(img_vertical_rgb)  # 显示垂直镜像图
ax4.imshow(img_diagonal_rgb)  # 显示对角镜像图
ax1.axis("off")  # 关闭左上角子图坐标轴
ax2.axis("off")  # 关闭右上角子图坐标轴
ax3.axis("off")  # 关闭左下角子图坐标轴
ax4.axis("off")  # 关闭右下角子图坐标轴
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.02, hspace=0.02)  # 手动控制四周和图像之间的间距，不使用tight_layout
plt.show()  # 显示最终3×3间隔布局图像
############################################################################

#######################################################################################3
# 14
# OpenCV算术运算如cv2.add()、cv2.subtract()会对像素值进行饱和处理，结果超过255时取255，小于0时取0。
# NumPy算术运算直接按数组数据类型计算，uint8类型超过范围时会发生溢出取模现象。
# OpenCV运算更适合图像处理中的像素计算，结果更符合图像显示效果。
# NumPy运算更适合普通矩阵和数组计算，但处理图像像素时需要注意数据范围。
###################################################################################

####################################################################3333
# 15
# cv2.imread()：读取图像文件，返回图像像素矩阵。
# cv2.imshow()：在窗口中显示图像。
# cv2.waitKey()：等待键盘按键，控制图像窗口停留时间。
# cv2.imwrite()：将图像保存到指定路径。
# cv2.resize()：改变图像大小。
# cv2.add()：实现图像加法运算。
# cv2.subtract()：实现图像减法运算。
# cv2.multiply()：实现图像乘法运算。
# cv2.divide()：实现图像除法运算。
# np.array()：创建或转换NumPy数组。
# np.dot()：实现矩阵乘法运算。
# cv2.addWeighted()：按照权重融合两幅图像。
# np.zeros()：创建全零数组，常用于生成掩膜图像。
# cv2.rectangle()：绘制矩形区域。
# cv2.bitwise_and()：实现按位与运算。
# cv2.bitwise_or()：实现按位或运算。
# cv2.bitwise_xor()：实现按位异或运算。
# cv2.bitwise_not()：实现按位非运算。
# cv2.warpAffine()：实现图像仿射变换，如平移和缩放。
# cv2.getRotationMatrix2D()：生成图像旋转矩阵。
# cv2.flip()：实现图像水平、垂直或对角镜像变换。
# cv2.destroyAllWindows()：关闭所有OpenCV显示窗口。
#######################################################################