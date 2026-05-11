import cv2  # 导入OpenCV库，用于图像读取、显示、保存和处理
import numpy as np  # 导入NumPy库，用于数组和矩阵运算
from pathlib import Path  # 导入Path类，用于跨平台拼接文件路径

path_main = Path(__file__).resolve().parents[1]  # 获取assignment1文件夹的绝对路径
resource_dir = path_main / "resource"  # 设置resource资源文件夹路径
result_dir = path_main / "result"  # 设置result结果文件夹路径
pics_dir = result_dir / "pics"  # 设置用于代替F盘pics文件夹的路径
ppics_dir = pics_dir / "ppics"  # 设置pics文件夹下ppics子目录的路径
pics_dir.mkdir(parents=True, exist_ok=True)  # 如果result/pics文件夹不存在，则自动创建
ppics_dir.mkdir(parents=True, exist_ok=True)  # 如果result/pics/ppics文件夹不存在，则自动创建

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
cv2.imshow("c2", imgCv2Add)  # 显示OpenCV加法得到的图像
imgNpAdd = np.array(imgScale + imgScale)  # 使用NumPy数组加法对图像进行相加
cv2.imshow("np", imgNpAdd)  # 显示NumPy加法得到的图像
cv2.waitKey(0)  # 等待键盘按键后继续执行程序

imgCv2diff = cv2.subtract(imgCv2Add, imgScale)  # 使用OpenCV减法计算图像差值
imgNpdiff = np.array(imgScale - imgNpAdd)  # 使用NumPy数组减法计算图像差值
cv2.imshow("c", imgCv2diff)  # 显示OpenCV减法结果
cv2.imshow("n", imgNpdiff)  # 显示NumPy减法结果
cv2.waitKey(0)  # 等待键盘按键后继续执行程序

imgGray = cv2.cvtColor(imgScale, cv2.COLOR_BGR2GRAY)  # 将缩放后的图像转换为灰度图
imgCv2mp = cv2.multiply(imgScale, imgScale)  # 使用OpenCV乘法对图像进行逐像素相乘
imgNpdot = np.dot(imgGray, imgGray)  # 使用NumPy的dot()函数进行矩阵乘法
cv2.imshow("c", imgCv2mp)  # 显示OpenCV乘法结果
cv2.imshow("dot", imgNpdot)  # 显示NumPy矩阵乘法结果
cv2.waitKey(0)  # 等待键盘按键后继续执行程序

den = (imgScale * 0.01).astype(np.uint8)  # 构造除法运算中的除数图像
den[den == 0] = 1  # 将除数中的0改为1，避免除以0
imgdivide = cv2.divide(imgScale, den)  # 使用cv2.divide()函数实现图像除法运算
cv2.imshow("div", imgdivide)  # 显示图像除法结果
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
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
cv2.imshow("original", img8)  # 显示原始图像
cv2.imshow("mask", mask)  # 显示掩膜图像
cv2.imshow("bitwise_and_keep_mask", img_and)  # 显示按位与结果
cv2.imshow("bitwise_or_remove_mask", img_or)  # 显示按位或结果
cv2.imshow("bitwise_xor", img_xor)  # 显示按位异或结果
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
M_move = np.array([[1, 0, 60], [0, 1, 40]], dtype=np.float32)  # 设置仿射变换矩阵，实现向右60像素和向下40像素平移
dst_same = cv2.warpAffine(img8, M_move, (200, 200))  # 使用cv2.warpAffine()实现显示窗口大小不变的平移
dst_large = cv2.warpAffine(img8, M_move, (260, 240))  # 使用cv2.warpAffine()实现显示窗口大小改变的平移
cv2.imshow("dst_same", dst_same)  # 显示窗口大小不变的平移结果
cv2.imshow("dst_large", dst_large)  # 显示窗口大小改变的平移结果
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
############################################################################

#######################################################################
# 11
img11 = imgNai  # 使用已经读取好的自拍图像
h, w = img11.shape[:2]  # 获取图像的高度和宽度
scale_x = 0.5  # 设置水平方向缩放比例
scale_y = 0.5  # 设置垂直方向缩放比例
M_scale_ratio = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])  # 构造按比例缩放的仿射变换矩阵
img_scale_ratio = cv2.warpAffine(img11, M_scale_ratio, (int(w * scale_x), int(h * scale_y)))  # 使用warpAffine按缩放比例缩放图像
target_w = 300  # 设置目标图像宽度
target_h = 300  # 设置目标图像高度
scale_x2 = target_w / w  # 根据目标宽度计算水平方向缩放比例
scale_y2 = target_h / h  # 根据目标高度计算垂直方向缩放比例
M_scale_size = np.float32([[scale_x2, 0, 0], [0, scale_y2, 0]])  # 构造按指定大小缩放的仿射变换矩阵
img_scale_size = cv2.warpAffine(img11, M_scale_size, (target_w, target_h))  # 使用warpAffine将图像缩放到指定大小
center = (w // 2, h // 2)  # 设置旋转中心为图像中心
M_clockwise = cv2.getRotationMatrix2D(center, -45, 1.0)  # 构造顺时针旋转45度的旋转矩阵
img_clockwise = cv2.warpAffine(img11, M_clockwise, (w, h))  # 使用warpAffine实现顺时针旋转
M_counterclockwise = cv2.getRotationMatrix2D(center, 45, 1.0)  # 构造逆时针旋转45度的旋转矩阵
img_counterclockwise = cv2.warpAffine(img11, M_counterclockwise, (w, h))  # 使用warpAffine实现逆时针旋转
cv2.imshow("original", img11)  # 显示原图
cv2.imshow("scale_ratio", img_scale_ratio)  # 显示按缩放比例缩放后的图像
cv2.imshow("scale_size", img_scale_size)  # 显示按指定大小缩放后的图像
cv2.imshow("clockwise", img_clockwise)  # 显示顺时针旋转后的图像
cv2.imshow("counterclockwise", img_counterclockwise)  # 显示逆时针旋转后的图像
cv2.waitKey(0)  # 等待键盘按键后继续执行程序
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
#####################################################################################################

########################################################################
# 12
img_path = pics_dir / "self.jpg"  # 设置自拍照片路径为assignment1/result/pics/self.jpg
if not img_path.exists():  # 判断自拍照片是否还没有放到pics文件夹中
    cv2.imwrite(str(img_path), imgNai)  # 如果没有自拍照片，则使用已读取的自拍图像生成self.jpg保证程序可执行
img12 = cv2.imread(str(img_path))  # 读取自拍图像
if img12 is None:  # 判断图像是否读取失败
    print(f"自拍图像读取失败，请检查路径：{img_path}")  # 输出错误提示和自拍图像路径
else:  # 图像读取成功后执行
    print("原图属性：")  # 输出原图属性标题
    print("原图尺寸：", img12.shape)  # 输出原图尺寸，包括高度、宽度和通道数
    print("原图高度：", img12.shape[0])  # 输出原图高度
    print("原图宽度：", img12.shape[1])  # 输出原图宽度
    print("原图通道数：", img12.shape[2])  # 输出原图通道数
    print("原图数据类型：", img12.dtype)  # 输出原图数据类型
    h, w = img12.shape[:2]  # 获取原图高度和宽度
    img_crop = img12[:, :w // 2]  # 将原图像规则剪裁一半，这里保留左半部分
    save_dir = ppics_dir  # 设置剪裁后图像保存目录为assignment1/result/pics/ppics
    save_dir.mkdir(parents=True, exist_ok=True)  # 如果ppics子目录不存在，则创建
    save_path = save_dir / "p_self.jpg"  # 设置剪裁后图像保存路径
    cv2.imwrite(str(save_path), img_crop)  # 将剪裁后的图像保存到ppics子目录中
    img_crop_read = cv2.imread(str(save_path))  # 重新读取保存后的剪裁图像
    print("剪裁后图像属性：")  # 输出剪裁后图像属性标题
    print("剪裁后图像尺寸：", img_crop_read.shape)  # 输出剪裁后图像尺寸
    print("剪裁后图像高度：", img_crop_read.shape[0])  # 输出剪裁后图像高度
    print("剪裁后图像宽度：", img_crop_read.shape[1])  # 输出剪裁后图像宽度
    print("剪裁后图像通道数：", img_crop_read.shape[2])  # 输出剪裁后图像通道数
    print("剪裁后图像数据类型：", img_crop_read.dtype)  # 输出剪裁后图像数据类型
    cv2.imshow("p_38", img_crop_read)  # 在名为p_38的窗口中显示剪裁后的图像
    cv2.waitKey(0)  # 等待键盘按键后继续执行程序
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
############################################################################################

###################################################
# 13
img_path = pics_dir / "self.jpg"  # 设置自拍照片路径为assignment1/result/pics/self.jpg
img13 = cv2.imread(str(img_path))  # 读取自拍图像
if img13 is None:  # 判断图像是否读取失败
    print(f"自拍图像读取失败，请检查路径：{img_path}")  # 输出错误提示和自拍图像路径
else:  # 图像读取成功后执行
    img_flip_horizontal = cv2.flip(img13, 1)  # 使用cv2.flip函数实现水平镜像变换
    img_flip_vertical = cv2.flip(img13, 0)  # 使用cv2.flip函数实现垂直镜像变换
    img_flip_diagonal = cv2.flip(img13, -1)  # 使用cv2.flip函数实现对角镜像变换
    cv2.imshow("original", img13)  # 显示原始自拍图像
    cv2.imshow("horizontal_flip", img_flip_horizontal)  # 显示水平镜像图像
    cv2.imshow("vertical_flip", img_flip_vertical)  # 显示垂直镜像图像
    cv2.imshow("diagonal_flip", img_flip_diagonal)  # 显示对角镜像图像
    cv2.waitKey(0)  # 等待键盘按键后继续执行程序
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
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