import cv2
import numpy as np
from pathlib import Path

path_main=Path.cwd()/"Assignment"/"assignment1"

##########################
# 1
img=cv2.imread(str(path_main/"resource"/"PostgreSQL.jpg"))
print(img)
##########################


################################
#2
cv2.imshow("img",img)
cv2.waitKey()
###################################

#####################################
# #3
# cv2.imwrite("F:\\pics\\Postgre.jpg")
# ########################################

# #####################################
# #4
# src_dir = Path("F:/pics")         # 设置原图片所在文件夹，这里是 F 盘下的 pics 文件夹

# dst_dir = src_dir / "ppics"       # 设置保存图片的目标文件夹，即 F:/pics/ppics

# dst_dir.mkdir(exist_ok=True)      # 如果 ppics 文件夹不存在，则创建；如果已存在，则不报错

# img_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  
# # 定义允许读取的图片后缀类型，避免读取到非图片文件

# for img_path in src_dir.iterdir():    
#     # 遍历 F:/pics 文件夹下的所有文件和文件夹

#     if not img_path.is_file():        
#         # 如果当前路径不是文件，比如是文件夹，则跳过

#         continue                      
#         # 跳过本次循环，继续检查下一个路径

#     if img_path.suffix.lower() not in img_suffix:  
#         # 判断当前文件的后缀是否属于图片格式，lower() 用于统一转换成小写

#         continue                      
#         # 如果不是图片文件，则跳过

#     img = cv2.imread(str(img_path))   
#     # 使用 cv2.imread() 读取图片，Path 对象需要转换成字符串

#     if img is None:                   
#         # 如果图片读取失败，img 会是 None

#         print(f"图片读取失败：{img_path}")  
#         # 输出读取失败的图片路径，方便检查问题

#         continue                      
#         # 跳过读取失败的图片

#     cv2.imshow("image", img)          
#     # 使用 cv2.imshow() 显示读取到的图片

#     save_path = dst_dir / img_path.name  
#     # 设置保存路径，文件名与原图片文件名保持一致

#     cv2.imwrite(str(save_path), img)  
#     # 使用 cv2.imwrite() 将图片保存到 bpics 文件夹中
#########################################################

######################################################
#5
print(img.shape[0:2])
print(img.shape[2])
print(img.dtype)
#########################################################


#######################################################
#6
imgScale=cv2.resize(img,(200,200))
imgCv2Add=cv2.add(imgScale,imgScale)
cv2.imshow('c2',imgCv2Add)
imgNpAdd=np.array(imgScale+imgScale)
cv2.imshow('np',imgNpAdd)
cv2.waitKey()

imgCv2diff=cv2.subtract(imgCv2Add,imgScale)
imgNpdiff=np.array(imgScale-imgNpAdd)
cv2.imshow('c',imgCv2diff)
cv2.imshow('n',imgNpdiff)
cv2.waitKey()

imgGray=cv2.cvtColor(imgScale,cv2.COLOR_BGR2GRAY)
imgCv2mp=cv2.multiply(imgScale,imgScale)
imgNpdot=np.dot(imgGray,imgGray)
cv2.imshow('c',imgCv2mp)
cv2.imshow('dot',imgNpdot)
cv2.waitKey()

den = (imgScale * 0.01).astype(np.uint8)   # 构造除数图像
den[den == 0] = 1                         # 避免除以 0
imgdivide = cv2.divide(imgScale, den)     # 使用 cv2.divide() 做图像除法
cv2.imshow("div", imgdivide)
cv2.waitKey(0)

###############################################

###################################################
#7
imgNai=cv2.imread(str(path_main/"resource"/"nailong.png"))
img2=cv2.resize(imgNai,(200,200))
res = cv2.addWeighted(imgScale, 0.7,img2 ,0.3, gamma=3)
cv2.imshow('res',res)
cv2.waitKey()
#####################################################

#########################################################
#8
img8 = cv2.resize(img, (200, 200))  # 将原图缩放为200×200，方便显示和处理
h, w = img8.shape[:2]  # 获取图像的高度和宽度
mask = np.zeros((h, w), dtype=np.uint8)  # 构造一个与原图大小相同的全黑单通道掩膜
cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)  # 在掩膜中画一个白色矩形区域，表示掩膜区域
img_and = cv2.bitwise_and(img8, img8, mask=mask)  # 使用按位与运算，只保留掩膜白色区域内的图像
mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 将单通道掩膜转换为三通道掩膜，方便与BGR图像进行按位运算
img_or = cv2.bitwise_or(img8, mask3)  # 使用按位或运算，将掩膜区域变成白色，相当于去掉掩膜内的图像
img_xor = cv2.bitwise_xor(img8, mask3)  # 使用按位异或运算，对掩膜区域内的图像进行异或处理
cv2.imshow("original", img8)  # 显示原始图像
cv2.imshow("mask", mask)  # 显示掩膜图像
cv2.imshow("bitwise_and_keep_mask", img_and)  # 显示按位与结果
cv2.imshow("bitwise_or_remove_mask", img_or)  # 显示按位或结果
cv2.imshow("bitwise_xor", img_xor)  # 显示按位异或结果
cv2.waitKey(0)  # 等待键盘按键
#################################################################################


#######################################################
#9
img_not=cv2.bitwise_not(img8)
cv2.imshow('bit',img_not)
cv2.waitKey()
################################################################

#######################################################################
#10
cv2.destroyAllWindows()
dst=cv2.warpAffine(img8,np.array([[1,0,60],
                                  [0,1,40]],dtype=np.float32),
                                  (200,200))
cv2.imshow('dst',dst)
cv2.waitKey()
############################################################################

#######################################################################
# 11
img11 = imgNai  # 使用已经读取好的自拍图像
h, w = img11.shape[:2]  # 获取图像的高度和宽度
scale_x = 0.5  # 设置水平方向缩放比例
scale_y = 0.5  # 设置垂直方向缩放比例
M_scale_ratio = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])  # 构造按比例缩放的仿射变换矩阵
img_scale_ratio = cv2.warpAffine(img11, M_scale_ratio, (int(w * scale_x), int(h * scale_y)))  # 使用warpAffine按缩放比例缩放图像
target_w = 300  # 直接设置目标图像宽度
target_h = 300  # 直接设置目标图像高度
scale_x2 = target_w / w  # 根据目标宽度计算水平方向缩放比例
scale_y2 = target_h / h  # 根据目标高度计算垂直方向缩放比例
M_scale_size = np.float32([[scale_x2, 0, 0], [0, scale_y2, 0]])  # 构造按指定大小缩放的仿射变换矩阵
img_scale_size = cv2.warpAffine(img11, M_scale_size, (target_w, target_h))  # 使用warpAffine将图像缩放到指定大小
center = (w // 2, h // 2)  # 设置旋转中心为图像中心
M_clockwise = cv2.getRotationMatrix2D(center, -45, 1.0)  # 构造顺时针旋转45度的旋转矩阵，负角度表示顺时针
img_clockwise = cv2.warpAffine(img11, M_clockwise, (w, h))  # 使用warpAffine实现顺时针旋转
M_counterclockwise = cv2.getRotationMatrix2D(center, 45, 1.0)  # 构造逆时针旋转45度的旋转矩阵，正角度表示逆时针
img_counterclockwise = cv2.warpAffine(img11, M_counterclockwise, (w, h))  # 使用warpAffine实现逆时针旋转
cv2.imshow("original", img11)  # 显示原图
cv2.imshow("scale_ratio", img_scale_ratio)  # 显示按缩放比例缩放后的图像
cv2.imshow("scale_size", img_scale_size)  # 显示按指定大小缩放后的图像
cv2.imshow("clockwise", img_clockwise)  # 显示顺时针旋转后的图像
cv2.imshow("counterclockwise", img_counterclockwise)  # 显示逆时针旋转后的图像
cv2.waitKey(0)  # 等待键盘按键
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
#####################################################################################################3

########################################################################33
# 12
img_path = Path("F:/pics/self.jpg")  # 设置自拍照片路径
img12 = cv2.imread(str(img_path))  # 读取自拍图像
if img12 is None:  # 判断图像是否读取失败
    print("自拍图像读取失败，请检查路径")  # 输出错误提示
else:  # 图像读取成功后执行
    print("原图属性：")  # 输出原图属性标题
    print("原图尺寸：", img12.shape)  # 输出原图尺寸，高度、宽度、通道数
    print("原图高度：", img12.shape[0])  # 输出原图高度
    print("原图宽度：", img12.shape[1])  # 输出原图宽度
    print("原图通道数：", img12.shape[2])  # 输出原图通道数
    print("原图数据类型：", img12.dtype)  # 输出原图数据类型
    h, w = img12.shape[:2]  # 获取原图高度和宽度
    img_crop = img12[:, :w // 2]  # 将原图像规则剪裁一半，这里保留左半部分
    save_dir = Path("F:/pics/ppics")  # 设置剪裁后图像保存目录
    save_dir.mkdir(exist_ok=True)  # 如果ppics子目录不存在，则创建
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
    cv2.waitKey(0)  # 等待键盘按键
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
############################################################################################

###################################################
# 13
img_path = Path("F:/pics/self.jpg")  # 设置自拍照片路径
img13 = cv2.imread(str(img_path))  # 读取自拍图像
if img13 is None:  # 判断图像是否读取失败
    print("自拍图像读取失败，请检查路径")  # 输出错误提示
else:  # 图像读取成功后执行
    img_flip_horizontal = cv2.flip(img13, 1)  # 水平镜像变换，flipCode为1表示左右翻转
    img_flip_vertical = cv2.flip(img13, 0)  # 垂直镜像变换，flipCode为0表示上下翻转
    img_flip_diagonal = cv2.flip(img13, -1)  # 对角镜像变换，flipCode为-1表示水平和垂直同时翻转
    cv2.imshow("original", img13)  # 显示原始自拍图像
    cv2.imshow("horizontal_flip", img_flip_horizontal)  # 显示水平镜像图像
    cv2.imshow("vertical_flip", img_flip_vertical)  # 显示垂直镜像图像
    cv2.imshow("diagonal_flip", img_flip_diagonal)  # 显示对角镜像图像
    cv2.waitKey(0)  # 等待键盘按键
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
############################################################################