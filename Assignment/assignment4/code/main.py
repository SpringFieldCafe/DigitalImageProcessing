import os  # 导入os模块，用于处理文件路径、文件夹等系统相关操作
import sys  # 导入sys模块，用于获取解释器信息或控制标准输入输出
import numpy as np  # 导入NumPy库，用于数组、矩阵和数值计算
from matplotlib import pyplot as plt  # 导入matplotlib绘图库，用于图像显示和结果可视化
import cv2  # 导入OpenCV库，用于图像读取、处理、显示和保存
from pathlib import Path  # 导入Path类，用于更方便地进行跨平台路径拼接和管理

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

code_dir=os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
assignment4_dir=Path(os.path.dirname(code_dir))
resource_dir=Path(assignment4_dir/"resource")
result_dir=Path(assignment4_dir/"result")
湖大logo_path=Path(resource_dir/"湖北大学logo")
self_image_path=Path(resource_dir/"self-image.jpg")
self_image2_path=Path(resource_dir/"self-image2.jpg")
test_plate_path=Path(resource_dir/"license plate"/"test_plate.jpg")

def cv_imread_chinese(path):  # 定义支持中文路径的图像读取函数
    img_array=np.fromfile(str(path),dtype=np.uint8)  # 使用numpy从文件路径中读取二进制数据
    img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)  # 使用OpenCV将二进制数据解码为彩色图像
    return img  # 返回读取到的图像

def cv_imwrite_chinese(path,img):  # 定义支持中文路径的图像保存函数
    ext=Path(path).suffix  # 获取保存文件的扩展名
    result,img_encode=cv2.imencode(ext,img)  # 按照扩展名对图像进行编码
    if result:  # 判断图像编码是否成功
        img_encode.tofile(str(path))  # 将编码后的图像数据写入指定路径
    return result  # 返回保存结果

def add_text(img,text):  # 定义给图像添加标题文字的辅助函数
    temp=img.copy()  # 复制输入图像，避免修改原图
    cv2.putText(temp,text,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,255),3)  # 在图像左上角添加紫色说明文字
    return temp  # 返回添加文字后的图像

def add_watermark(self_img,logo_resize,logo_mask,watermark_alpha,margin):  # 定义添加透明水印的辅助函数
    self_h,self_w=self_img.shape[:2]  # 获取自拍图像的高度和宽度
    new_logo_h,new_logo_w=logo_resize.shape[:2]  # 获取缩放后logo的高度和宽度
    watermark_img=self_img.copy().astype(np.float32)  # 复制自拍图像并转换为浮点型，便于透明融合计算
    x1=self_w-new_logo_w-margin  # 计算logo左上角x坐标
    y1=self_h-new_logo_h-margin  # 计算logo左上角y坐标
    x2=x1+new_logo_w  # 计算logo右下角x坐标
    y2=y1+new_logo_h  # 计算logo右下角y坐标
    if x1<0 or y1<0:  # 判断logo是否超出自拍图像范围
        raise ValueError("logo尺寸过大，请调小 logo_width_ratio")  # 如果logo尺寸过大，则抛出错误提示
    roi=watermark_img[y1:y2,x1:x2]  # 截取自拍图像中用于添加水印的区域
    logo_float=logo_resize.astype(np.float32)  # 将缩放后的logo转换为浮点型
    alpha=(logo_mask.astype(np.float32)/255.0)*watermark_alpha  # 根据logo掩膜和透明度参数计算真实透明度
    alpha_3=cv2.merge([alpha,alpha,alpha])  # 将单通道透明度扩展为三通道透明度
    roi_watermark=roi*(1-alpha_3)+logo_float*alpha_3  # 使用透明融合公式生成带水印的区域
    watermark_img[y1:y2,x1:x2]=roi_watermark  # 将融合后的水印区域放回原图对应位置
    watermark_img=np.clip(watermark_img,0,255).astype(np.uint8)  # 将图像像素限制到0到255并转换为uint8类型
    return watermark_img  # 返回添加水印后的图像

##############################################
#1
code_dir=Path(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))  # 获取当前代码文件所在文件夹路径
assignment4_dir=code_dir.parent  # 获取assignment4目录路径
resource_dir=assignment4_dir/"resource"  # 设置resource资源文件夹路径
result_dir=assignment4_dir/"result"  # 设置result结果输出文件夹路径
result_dir.mkdir(parents=True,exist_ok=True)  # 如果result文件夹不存在，则自动创建
湖大logo_path=resource_dir/"湖北大学logo.jpg"  # 设置湖大logo图片路径
self_image_path=resource_dir/"self-image.jpg"  # 设置自拍图片路径
self_img=cv_imread_chinese(self_image_path)  # 读取自拍图像
logo_img=cv_imread_chinese(湖大logo_path)  # 读取湖大logo图像
if self_img is None:  # 判断自拍图片是否读取失败
    raise FileNotFoundError(f"未找到自拍图片：{self_image_path}")  # 如果读取失败，则抛出文件不存在错误
if logo_img is None:  # 判断湖大logo图片是否读取失败
    raise FileNotFoundError(f"未找到湖大logo图片：{湖大logo_path}")  # 如果读取失败，则抛出文件不存在错误
white_threshold=245  # 设置白色背景阈值，越接近255，去白背景越严格
alpha_hidden=0.18  # 设置隐藏水印透明度版本
alpha_visible=0.8  # 设置明显水印透明度版本
logo_width_ratio=0.7 # 设置logo宽度占自拍图宽度的比例，数值越大水印越大
margin=30  # 设置logo距离右下角边缘的距离
self_h,self_w=self_img.shape[:2]  # 获取自拍图像的高度和宽度
logo_h,logo_w=logo_img.shape[:2]  # 获取logo图像的高度和宽度
new_logo_w=int(self_w*logo_width_ratio)  # 根据自拍图宽度比例计算缩放后的logo宽度
new_logo_h=int(logo_h*new_logo_w/logo_w)  # 根据原logo宽高比例计算缩放后的logo高度
logo_resize=cv2.resize(logo_img,(new_logo_w,new_logo_h),interpolation=cv2.INTER_AREA)  # 对logo进行放大或缩小处理
logo_gray=cv2.cvtColor(logo_resize,cv2.COLOR_BGR2GRAY)  # 将缩放后的logo转换为灰度图
_,white_mask=cv2.threshold(logo_gray,white_threshold,255,cv2.THRESH_BINARY)  # 根据阈值提取logo中的白色背景区域
logo_mask=cv2.bitwise_not(white_mask)  # 对白色背景掩膜取反，得到logo主体区域
b,g,r=cv2.split(logo_resize)  # 将缩放后的logo拆分为B、G、R三个通道
transparent_logo=cv2.merge([b,g,r,logo_mask])  # 将B、G、R和透明度掩膜合并为BGRA透明图像
watermark_hidden=add_watermark(self_img,logo_resize,logo_mask,alpha_hidden,margin)  # 生成alpha为0.18的隐藏水印结果
watermark_visible=add_watermark(self_img,logo_resize,logo_mask,alpha_visible,margin)  # 生成alpha为0.8的明显水印结果
diff_hidden=cv2.absdiff(self_img,watermark_hidden)  # 计算原图与隐藏水印图之间的差异
diff_hidden_gray=cv2.cvtColor(diff_hidden,cv2.COLOR_BGR2GRAY)  # 将隐藏水印差异图转换为灰度图
diff_hidden_enhance=cv2.convertScaleAbs(diff_hidden_gray,alpha=30,beta=0)  # 放大隐藏水印差异，便于观察alpha为0.18的水印轮廓
diff_hidden_enhance_color=cv2.cvtColor(diff_hidden_enhance,cv2.COLOR_GRAY2BGR)  # 将增强差异灰度图转换为三通道图像
cv_imwrite_chinese(result_dir/"第一题_透明湖大logo.png",transparent_logo)  # 保存去白底后的透明logo图像
cv_imwrite_chinese(result_dir/"第一题_alpha_0.18_隐藏水印.png",watermark_hidden)  # 保存alpha为0.18的隐藏水印结果图
cv_imwrite_chinese(result_dir/"第一题_alpha_0.8_明显水印.png",watermark_visible)  # 保存alpha为0.8的明显水印结果图
cv_imwrite_chinese(result_dir/"第一题_alpha_0.18_差分增强查看.png",diff_hidden_enhance_color)  # 保存alpha为0.18的差分增强查看图
stack_compare=stackImages(0.7,[[add_text(self_img,"Original Image"),add_text(watermark_hidden,"Alpha 0.18 Hidden Watermark")],[add_text(watermark_visible,"Alpha 0.8 Visible Watermark"),add_text(diff_hidden_enhance_color,"Alpha 0.18 Difference x30")]])  # 将原图、隐藏水印、明显水印和差分增强图进行2乘2堆叠
cv_imwrite_chinese(result_dir/"第一题_两种透明度水印堆叠对比.png",stack_compare)  # 保存两种透明度水印堆叠对比图
plt.figure(figsize=(16,10))  # 创建matplotlib可视化窗口并设置图像大小
plt.subplot(2,3,1)  # 创建2行3列的第1个子图
plt.imshow(cv2.cvtColor(self_img,cv2.COLOR_BGR2RGB))  # 显示原始自拍图像，并将BGR转换为RGB
plt.title("Original Self Image")  # 设置第1个子图标题
plt.axis("off")  # 关闭第1个子图坐标轴
plt.subplot(2,3,2)  # 创建2行3列的第2个子图
plt.imshow(cv2.cvtColor(logo_resize,cv2.COLOR_BGR2RGB))  # 显示放大后的湖大logo图像，并将BGR转换为RGB
plt.title(f"Resized Logo ratio={logo_width_ratio}")  # 设置第2个子图标题并显示logo缩放比例
plt.axis("off")  # 关闭第2个子图坐标轴
plt.subplot(2,3,3)  # 创建2行3列的第3个子图
plt.imshow(logo_mask,cmap="gray")  # 显示只通过阈值获得的logo主体掩膜
plt.title(f"Logo Mask threshold={white_threshold}")  # 设置第3个子图标题并显示阈值参数
plt.axis("off")  # 关闭第3个子图坐标轴
plt.subplot(2,3,4)  # 创建2行3列的第4个子图
plt.imshow(cv2.cvtColor(watermark_hidden,cv2.COLOR_BGR2RGB))  # 显示alpha为0.18的隐藏水印结果图
plt.title(f"Watermark alpha={alpha_hidden}")  # 设置第4个子图标题并显示隐藏水印透明度
plt.axis("off")  # 关闭第4个子图坐标轴
plt.subplot(2,3,5)  # 创建2行3列的第5个子图
plt.imshow(cv2.cvtColor(watermark_visible,cv2.COLOR_BGR2RGB))  # 显示alpha为0.8的明显水印结果图
plt.title(f"Watermark alpha={alpha_visible}")  # 设置第5个子图标题并显示明显水印透明度
plt.axis("off")  # 关闭第5个子图坐标轴
plt.subplot(2,3,6)  # 创建2行3列的第6个子图
plt.imshow(diff_hidden_enhance,cmap="gray")  # 显示alpha为0.18的差分增强查看图
plt.title("Alpha 0.18 Difference x30")  # 设置第6个子图标题
plt.axis("off")  # 关闭第6个子图坐标轴
plt.tight_layout()  # 自动调整子图布局，避免内容重叠
plt.savefig(result_dir/"第一题_alpha_0.18和0.8同图对比figure.png",dpi=200,bbox_inches="tight")  # 保存alpha为0.18和0.8的同图对比figure
plt.show()  # 显示matplotlib可视化结果
cv2.imshow("Question 1 Alpha Compare",stack_compare)  # 使用OpenCV窗口显示两种透明度水印堆叠对比图
cv2.waitKey(0)  # 等待键盘按键
cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
#############################################

############################################################
#2
#=======================  自定义插值辅助函数  ===================
def nearest_neighbor_resize_custom(img,scale):  # 定义最近邻插值放大函数
    src_h,src_w,channels=img.shape  # 获取原图高度、宽度和通道数
    dst_h=int(src_h*scale)  # 根据放大倍数计算目标图像高度
    dst_w=int(src_w*scale)  # 根据放大倍数计算目标图像宽度
    y_index=np.round(np.arange(dst_h)/scale).astype(np.int32)  # 计算目标图像每一行对应的原图最近邻行坐标
    x_index=np.round(np.arange(dst_w)/scale).astype(np.int32)  # 计算目标图像每一列对应的原图最近邻列坐标
    y_index=np.clip(y_index,0,src_h-1)  # 限制行坐标范围，防止越界
    x_index=np.clip(x_index,0,src_w-1)  # 限制列坐标范围，防止越界
    dst_img=img[y_index[:,None],x_index[None,:]]  # 使用NumPy索引一次性完成最近邻插值映射
    return dst_img.copy()  # 返回最近邻插值放大后的图像
def bilinear_resize_custom(img,scale):  # 定义双线性插值放大函数
    src_h,src_w,channels=img.shape  # 获取原图高度、宽度和通道数
    dst_h=int(src_h*scale)  # 根据放大倍数计算目标图像高度
    dst_w=int(src_w*scale)  # 根据放大倍数计算目标图像宽度
    dst_img=np.zeros((dst_h,dst_w,channels),dtype=np.uint8)  # 创建目标图像空数组
    x=np.arange(dst_w)/scale  # 计算目标图像所有列对应的原图浮点x坐标
    x0=np.floor(x).astype(np.int32)  # 获取左侧像素点x坐标
    x1=np.clip(x0+1,0,src_w-1)  # 获取右侧像素点x坐标并防止越界
    x0=np.clip(x0,0,src_w-1)  # 限制左侧像素点x坐标范围
    dx=(x-x0).astype(np.float32)  # 计算x方向上的小数距离
    dx_3=dx[:,None]  # 将x方向权重扩展为二维，便于和三通道像素计算
    for y in range(dst_h):  # 遍历目标图像每一行，但每一行内部使用NumPy向量化计算
        src_y=y/scale  # 根据目标图像y坐标反推原图中的浮点y坐标
        y0=int(np.floor(src_y))  # 获取上方像素点y坐标
        y1=min(y0+1,src_h-1)  # 获取下方像素点y坐标并防止越界
        y0=min(y0,src_h-1)  # 限制上方像素点y坐标范围
        dy=np.float32(src_y-y0)  # 计算y方向上的小数距离
        p00=img[y0,x0].astype(np.float32)  # 获取当前行对应的左上像素数组
        p01=img[y0,x1].astype(np.float32)  # 获取当前行对应的右上像素数组
        p10=img[y1,x0].astype(np.float32)  # 获取当前行对应的左下像素数组
        p11=img[y1,x1].astype(np.float32)  # 获取当前行对应的右下像素数组
        top=p00*(1-dx_3)+p01*dx_3  # 计算上方两个像素在x方向上的线性插值
        bottom=p10*(1-dx_3)+p11*dx_3  # 计算下方两个像素在x方向上的线性插值
        value=top*(1-dy)+bottom*dy  # 计算y方向上的线性插值
        dst_img[y]=np.clip(value,0,255).astype(np.uint8)  # 将当前行插值结果写入目标图像
    sys.stdout.flush()  # 立即刷新输出缓冲区
    return dst_img  # 返回双线性插值放大后的图像
def put_text_blurviolet(img,text):  # 定义使用blurviolet颜色添加说明文字的辅助函数
    temp=img.copy()  # 复制输入图像，避免直接修改原图
    blurviolet_color=(226,43,138)  # 设置OpenCV中的blurviolet近似颜色，BGR格式
    cv2.putText(temp,text,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,blurviolet_color,3)  # 在图像左上角添加blurviolet颜色说明文字
    return temp  # 返回添加文字后的图像


#=========================================================================================================
code_dir=Path(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))  # 获取当前代码文件所在文件夹路径
assignment4_dir=code_dir.parent  # 获取assignment4目录路径
resource_dir=assignment4_dir/"resource"  # 设置resource资源文件夹路径
result_dir=assignment4_dir/"result"  # 设置result结果输出文件夹路径
result_dir.mkdir(parents=True,exist_ok=True)  # 如果result文件夹不存在，则自动创建
self_image2_path=Path(resource_dir/"self-image2.jpg")  # 设置第二张自拍照片路径
self_img2=cv_imread_chinese(self_image2_path)  # 使用支持中文路径的函数读取第二张自拍图像
if self_img2 is None:  # 判断第二张自拍照片是否读取失败
    raise FileNotFoundError(f"未找到第二张自拍图片：{self_image2_path}")  # 如果读取失败，则抛出文件不存在错误
scale_factor=1.5  # 设置图像放大倍数为1.5倍
original_h,original_w=self_img2.shape[:2]  # 获取原始自拍图像的高度和宽度
target_h=int(original_h*scale_factor)  # 根据放大倍数计算目标图像高度
target_w=int(original_w*scale_factor)  # 根据放大倍数计算目标图像宽度
nearest_img=nearest_neighbor_resize_custom(self_img2,scale_factor)  # 调用自定义最近邻插值函数将自拍图像放大1.5倍
bilinear_img=bilinear_resize_custom(self_img2,scale_factor)  # 调用自定义双线性插值函数将自拍图像放大1.5倍
cv_imwrite_chinese(result_dir/"第二题_最近邻插值放大1.5倍.png",nearest_img)  # 保存最近邻插值放大结果图像
cv_imwrite_chinese(result_dir/"第二题_双线性插值放大1.5倍.png",bilinear_img)  # 保存双线性插值放大结果图像
original_text=put_text_blurviolet(self_img2,"Original Self Image 2")  # 给原始自拍图像添加blurviolet颜色说明文字
nearest_text=put_text_blurviolet(nearest_img,"Nearest Neighbor x1.5")  # 给最近邻插值结果添加blurviolet颜色说明文字
bilinear_text=put_text_blurviolet(bilinear_img,"Bilinear Interpolation x1.5")  # 给双线性插值结果添加blurviolet颜色说明文字
cv_imwrite_chinese(result_dir/"第二题_原始自拍图像self-image2.png",original_text)  # 单独保存带文字说明的原始自拍图像
cv_imwrite_chinese(result_dir/"第二题_最近邻插值放大1.5倍_单图展示.png",nearest_text)  # 单独保存带文字说明的最近邻插值放大图像
cv_imwrite_chinese(result_dir/"第二题_双线性插值放大1.5倍_单图展示.png",bilinear_text)  # 单独保存带文字说明的双线性插值放大图像
plt.figure(figsize=(8,6))  # 创建matplotlib窗口用于显示原始自拍图像
plt.imshow(cv2.cvtColor(self_img2,cv2.COLOR_BGR2RGB))  # 显示原始自拍图像并将BGR格式转换为RGB格式
plt.title(f"Original Image\nsize={original_w}x{original_h}")  # 设置原始图像标题并显示原始尺寸
plt.axis("off")  # 关闭坐标轴
plt.tight_layout()  # 自动调整图像布局
plt.savefig(result_dir/"第二题_原始自拍图像单独figure.png",dpi=200,bbox_inches="tight")  # 单独保存原始自拍图像figure
plt.show()  # 显示原始自拍图像figure
plt.figure(figsize=(8,6))  # 创建matplotlib窗口用于显示最近邻插值结果
plt.imshow(cv2.cvtColor(nearest_img,cv2.COLOR_BGR2RGB))  # 显示最近邻插值放大图像并将BGR格式转换为RGB格式
plt.title(f"Nearest Neighbor Interpolation\nscale={scale_factor}, size={target_w}x{target_h}")  # 设置最近邻插值标题并显示放大倍数和目标尺寸
plt.axis("off")  # 关闭坐标轴
plt.tight_layout()  # 自动调整图像布局
plt.savefig(result_dir/"第二题_最近邻插值单独figure.png",dpi=200,bbox_inches="tight")  # 单独保存最近邻插值figure
plt.show()  # 显示最近邻插值figure
plt.figure(figsize=(8,6))  # 创建matplotlib窗口用于显示双线性插值结果
plt.imshow(cv2.cvtColor(bilinear_img,cv2.COLOR_BGR2RGB))  # 显示双线性插值放大图像并将BGR格式转换为RGB格式
plt.title(f"Bilinear Interpolation\nscale={scale_factor}, size={target_w}x{target_h}")  # 设置双线性插值标题并显示放大倍数和目标尺寸
plt.axis("off")  # 关闭坐标轴
plt.tight_layout()  # 自动调整图像布局
plt.savefig(result_dir/"第二题_双线性插值单独figure.png",dpi=200,bbox_inches="tight")  # 单独保存双线性插值figure
plt.show()  # 显示双线性插值figure
plt.figure(figsize=(9,6))  # 创建matplotlib窗口用于可视化重要参数
plt.text(0.05,0.85,f"Scale Factor: {scale_factor}",fontsize=14)  # 显示图像放大倍数参数
plt.text(0.05,0.68,f"Original Size: {original_w} x {original_h}",fontsize=14)  # 显示原始图像尺寸参数
plt.text(0.05,0.51,f"Target Size: {target_w} x {target_h}",fontsize=14)  # 显示目标图像尺寸参数
plt.text(0.05,0.34,"Nearest Neighbor: choose the closest source pixel",fontsize=12)  # 显示最近邻插值核心思想
plt.text(0.05,0.20,"Bilinear: weighted average of four neighboring pixels",fontsize=12)  # 显示双线性插值核心思想
plt.title("Question 2 Important Parameters")  # 设置重要参数说明图标题
plt.axis("off")  # 关闭坐标轴
plt.tight_layout()  # 自动调整图像布局
plt.savefig(result_dir/"第二题_插值放大重要参数单独figure.png",dpi=200,bbox_inches="tight")  # 单独保存重要参数可视化图
plt.show()  # 显示重要参数可视化图
sys.stdout.flush()  # 刷新标准输出缓冲区
cv2.imshow("Original Self Image 2",original_text)  # 单独使用OpenCV窗口显示原始自拍图像
cv2.waitKey(0)  # 等待键盘按键后继续显示下一张图
cv2.imshow("Nearest Neighbor x1.5",nearest_text)  # 单独使用OpenCV窗口显示最近邻插值结果图像
cv2.waitKey(0)  # 等待键盘按键后继续显示下一张图
cv2.imshow("Bilinear Interpolation x1.5",bilinear_text)  # 单独使用OpenCV窗口显示双线性插值结果图像
cv2.waitKey(0)  # 等待键盘按键
cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
######################################################################

###############################################
#3
# 自然彩色图像车牌提取的基本思路是：qqq批量读取车牌图像 → HSV蓝色分割 → 形态学处理 → 轮廓筛选 → 透视矩阵校正 → 车牌裁剪 → 字符区域二值化
plate_dir = resource_dir / "license plate"  # 设置车牌图片所在文件夹路径
plate_result_dir = result_dir / "license_plate_result"  # 设置车牌检测结果保存文件夹路径
plate_result_dir.mkdir(parents=True, exist_ok=True)  # 如果车牌检测结果文件夹不存在，则自动创建
plate_paths = sorted([p for p in plate_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])  # 循环获取车牌文件夹中所有常见格式图片路径
if len(plate_paths) == 0:  # 判断车牌文件夹中是否存在可读取的图片文件
    raise FileNotFoundError(f"未在该文件夹中找到车牌图片：{plate_dir}")  # 如果没有找到图片，则抛出文件不存在错误
def order_points(pts):  # 定义四点排序函数，用于透视变换前统一角点顺序
    rect = np.zeros((4, 2), dtype=np.float32)  # 创建4×2矩阵保存排序后的四个角点
    s = pts.sum(axis=1)  # 计算每个点横坐标和纵坐标之和
    diff = np.diff(pts, axis=1)  # 计算每个点纵坐标减横坐标的差值
    rect[0] = pts[np.argmin(s)]  # 坐标和最小的点作为左上角
    rect[2] = pts[np.argmax(s)]  # 坐标和最大的点作为右下角
    rect[1] = pts[np.argmin(diff)]  # 差值最小的点作为右上角
    rect[3] = pts[np.argmax(diff)]  # 差值最大的点作为左下角
    return rect  # 返回排序后的四个角点
def process_single_plate(imgTestPlate, plate_name):  # 定义单张车牌图像处理函数
    max_width = 900  # 设置最大处理宽度，避免原图过大影响显示和处理速度
    h0, w0 = imgTestPlate.shape[:2]  # 获取原始图像的高度和宽度
    if w0 > max_width:  # 判断原图宽度是否超过最大处理宽度
        scale = max_width / w0  # 根据最大宽度计算缩放比例
        imgTestPlate = cv2.resize(imgTestPlate, (max_width, int(h0 * scale)))  # 按比例缩小图像
    imgTestPlate_original = imgTestPlate.copy()  # 复制缩放后的原图用于后续展示和保存
    violet_color = (238, 130, 238)  # 设置文字颜色为Violet，OpenCV中使用BGR顺序
    imgHSV = cv2.cvtColor(imgTestPlate, cv2.COLOR_BGR2HSV)  # 将BGR图像转换到HSV色彩空间，便于提取蓝色车牌
    lower_blue = np.array([90, 50, 40], dtype=np.uint8)  # 设置蓝色车牌HSV阈值下限
    upper_blue = np.array([145, 255, 255], dtype=np.uint8)  # 设置蓝色车牌HSV阈值上限
    mask_blue = cv2.inRange(imgHSV, lower_blue, upper_blue)  # 根据HSV蓝色范围生成二值掩膜图
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))  # 创建横向矩形结构元素，用于连接车牌内部区域
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 创建5×5矩形结构元素，用于去除小噪声
    mask_close = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # 对蓝色掩膜做闭运算，连接断裂区域
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 对闭运算结果做开运算，去除零散噪声
    contours_info = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找二值图中的外部轮廓
    contours = contours_info[-2]  # 获取轮廓列表，兼容OpenCV3和OpenCV4
    best_box = None  # 初始化最佳车牌四角点
    best_score = 0  # 初始化最佳车牌候选区域得分
    for cnt in contours:  # 遍历所有候选轮廓
        area = cv2.contourArea(cnt)  # 计算当前轮廓面积
        if area < 800:  # 判断当前轮廓面积是否过小
            continue  # 跳过面积过小的轮廓
        peri = cv2.arcLength(cnt, True)  # 计算当前轮廓周长
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)  # 对轮廓进行多边形近似，尝试提取车牌四角点
        rect = cv2.minAreaRect(cnt)  # 获取当前轮廓的最小外接旋转矩形
        width = rect[1][0]  # 获取旋转矩形宽度
        height = rect[1][1]  # 获取旋转矩形高度
        if width == 0 or height == 0:  # 判断旋转矩形宽高是否有效
            continue  # 跳过宽高无效的候选区域
        if len(approx) == 4:  # 判断轮廓近似后是否刚好得到四个点
            box = approx.reshape(4, 2).astype(np.float32)  # 使用近似得到的四边形角点
            method_weight = 1.3  # 四边形角点更适合透视变换，因此提高候选得分权重
        else:  # 如果轮廓无法直接近似为四边形
            box = cv2.boxPoints(rect).astype(np.float32)  # 使用最小外接旋转矩形的四个角点作为备用
            method_weight = 1.0  # 备用角点得分权重保持正常
        ordered_box = order_points(box)  # 对候选四角点进行左上、右上、右下、左下排序
        tl, tr, br, bl = ordered_box  # 分别取出排序后的四个角点
        box_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2  # 计算候选区域平均宽度
        box_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2  # 计算候选区域平均高度
        if box_width == 0 or box_height == 0:  # 判断候选区域宽高是否有效
            continue  # 跳过宽高无效的候选区域
        ratio = max(box_width, box_height) / min(box_width, box_height)  # 计算候选区域长宽比
        if ratio < 2.0 or ratio > 6.8:  # 根据车牌扁长矩形特征筛选候选区域
            continue  # 跳过长宽比不合理的区域
        rect_area = width * height  # 计算最小外接旋转矩形面积
        fill_ratio = area / rect_area if rect_area > 0 else 0  # 计算轮廓面积与矩形面积的填充比例
        score = area * fill_ratio * method_weight  # 综合面积、填充比例和角点质量计算候选得分
        if score > best_score:  # 判断当前候选是否优于之前的最佳候选
            best_score = score  # 更新最佳候选得分
            best_box = ordered_box  # 更新最佳车牌四角点
    if best_box is None:  # 判断是否检测到车牌候选区域
        print(f"{plate_name}：未检测到车牌区域，已跳过")  # 输出当前图片未检测到车牌的提示
        fail_img = imgTestPlate_original.copy()  # 复制原图用于失败结果展示
        cv2.putText(fail_img, "Plate Not Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, violet_color, 3)  # 在失败图像上添加提示文字
        cv2.imwrite(str(plate_result_dir / f"{plate_name}_detect_failed.jpg"), fail_img)  # 保存检测失败的图像
        return fail_img  # 返回检测失败的展示图
    plate_rect = best_box.astype(np.float32)  # 将最佳车牌四角点转换为float32类型，便于计算透视矩阵
    draw_box = plate_rect.astype(np.int32)  # 将车牌四角点转换为整数，便于绘制轮廓框
    imgTestPlate_box = imgTestPlate.copy()  # 复制原图，用于绘制车牌定位结果
    cv2.drawContours(imgTestPlate_box, [draw_box], -1, (0, 255, 0), 3)  # 在原图上绘制检测到的车牌四边形区域
    cv2.putText(imgTestPlate_box, "License Plate ROI", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, violet_color, 3)  # 在定位图上添加车牌区域文字说明
    plate_std_width = 440  # 设置透视变换后车牌标准宽度
    plate_std_height = 140  # 设置透视变换后车牌标准高度
    dst_pts = np.array([[0, 0], [plate_std_width - 1, 0], [plate_std_width - 1, plate_std_height - 1], [0, plate_std_height - 1]], dtype=np.float32)  # 设置标准矩形目标四角点
    M = cv2.getPerspectiveTransform(plate_rect, dst_pts)  # 根据原车牌四角点和目标矩形四角点计算透视变换矩阵
    print(f"\n{plate_name} 的透视变换矩阵 M：")  # 输出当前图片名称和透视矩阵提示
    print(M)  # 打印3×3透视变换矩阵
    plate_warp = cv2.warpPerspective(imgTestPlate, M, (plate_std_width, plate_std_height))  # 使用透视矩阵将倾斜车牌校正为标准矩形
    plate_hsv = cv2.cvtColor(plate_warp, cv2.COLOR_BGR2HSV)  # 将校正后的车牌图转换到HSV色彩空间
    plate_mask = cv2.inRange(plate_hsv, lower_blue, upper_blue)  # 再次提取校正车牌中的蓝色区域
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel_open, iterations=1)  # 对校正车牌掩膜做闭运算，减少内部空洞
    plate_contours_info = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找校正车牌图中的蓝色主体轮廓
    plate_contours = plate_contours_info[-2]  # 获取校正车牌图中的轮廓列表
    if len(plate_contours) > 0:  # 判断校正车牌中是否还能找到蓝色主体区域
        plate_main = max(plate_contours, key=cv2.contourArea)  # 选择面积最大的蓝色区域作为车牌主体
        x, y, w, h = cv2.boundingRect(plate_main)  # 获取车牌主体区域的外接矩形
        pad = 5  # 设置裁剪边缘扩展像素，避免切掉车牌边缘
        x1 = max(x - pad, 0)  # 计算裁剪区域左边界
        y1 = max(y - pad, 0)  # 计算裁剪区域上边界
        x2 = min(x + w + pad, plate_warp.shape[1])  # 计算裁剪区域右边界
        y2 = min(y + h + pad, plate_warp.shape[0])  # 计算裁剪区域下边界
        plate_crop = plate_warp[y1:y2, x1:x2]  # 根据蓝色主体区域精细裁剪车牌
    else:  # 如果没有找到蓝色主体区域
        plate_crop = plate_warp.copy()  # 保留透视校正后的车牌图作为裁剪结果
    if plate_crop.size == 0:  # 判断裁剪图像是否为空
        plate_crop = plate_warp.copy()  # 如果裁剪失败，则使用透视校正后的车牌图
    plate_crop = cv2.resize(plate_crop, (440, 140))  # 将车牌区域统一拉伸到440×140
    margin_x = int(plate_crop.shape[1] * 0.04)  # 计算左右边框裁剪宽度
    margin_y = int(plate_crop.shape[0] * 0.08)  # 计算上下边框裁剪高度
    plate_character_area = plate_crop[margin_y:plate_crop.shape[0] - margin_y, margin_x:plate_crop.shape[1] - margin_x]  # 裁剪车牌中间字符区域
    plate_gray = cv2.cvtColor(plate_character_area, cv2.COLOR_BGR2GRAY)  # 将字符区域转换为灰度图
    plate_gray = cv2.equalizeHist(plate_gray)  # 对灰度图进行直方图均衡化，增强字符对比度
    plate_blur = cv2.GaussianBlur(plate_gray, (3, 3), 0)  # 对灰度图进行轻微高斯滤波，减少噪声
    _, plate_binary = cv2.threshold(plate_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 使用Otsu自动阈值法得到字符区域二值图
    one_result_dir = plate_result_dir / plate_name  # 为当前车牌图片创建单独结果文件夹路径
    one_result_dir.mkdir(parents=True, exist_ok=True)  # 如果当前图片结果文件夹不存在，则自动创建
    cv2.imwrite(str(one_result_dir / "01_original.jpg"), imgTestPlate_original)  # 保存原始车牌图
    cv2.imwrite(str(one_result_dir / "02_blue_mask.jpg"), mask_blue)  # 保存HSV蓝色掩膜图
    cv2.imwrite(str(one_result_dir / "03_detect_box.jpg"), imgTestPlate_box)  # 保存车牌定位框结果图
    cv2.imwrite(str(one_result_dir / "04_matrix_warp.jpg"), plate_warp)  # 保存透视矩阵校正后的车牌图
    cv2.imwrite(str(one_result_dir / "05_crop.jpg"), plate_crop)  # 保存精细裁剪后的车牌图
    cv2.imwrite(str(one_result_dir / "06_character_binary.jpg"), plate_binary)  # 保存字符区域二值图
    imgTestPlate_original_text = imgTestPlate_original.copy()  # 复制原图，便于添加文字
    mask_blue_text = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)  # 将蓝色掩膜图转换为三通道图像，便于添加文字
    plate_warp_text = plate_warp.copy()  # 复制透视校正图，便于添加文字
    plate_crop_text = plate_crop.copy()  # 复制裁剪结果图，便于添加文字
    plate_binary_text = cv2.cvtColor(plate_binary, cv2.COLOR_GRAY2BGR)  # 将二值图转换为三通道图像，便于添加文字
    cv2.putText(imgTestPlate_original_text, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, violet_color, 3)  # 在原图上添加说明文字
    cv2.putText(mask_blue_text, "HSV Blue Mask", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, violet_color, 3)  # 在蓝色掩膜图上添加说明文字
    cv2.putText(plate_warp_text, "Matrix Warp 440x140", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, violet_color, 3)  # 在透视矩阵拉伸图上添加说明文字
    cv2.putText(plate_crop_text, "Final Plate Crop", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, violet_color, 3)  # 在车牌裁剪图上添加说明文字
    cv2.putText(plate_binary_text, "Character Binary", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, violet_color, 3)  # 在字符二值图上添加说明文字
    plate_stack = stackImages(0.45, [[imgTestPlate_original_text, imgTestPlate_box, mask_blue_text], [plate_warp_text, plate_crop_text, plate_binary_text]])  # 将必要处理结果按2行3列堆叠显示
    cv2.imwrite(str(one_result_dir / "plate_process_stack.jpg"), plate_stack)  # 保存当前车牌处理流程堆叠图
    cv2.imwrite(str(plate_result_dir / f"{plate_name}_plate_process_stack.jpg"), plate_stack)  # 在总结果文件夹中保存当前车牌流程堆叠图
    return plate_stack  # 返回当前车牌的处理流程堆叠图
all_detect_results = []  # 创建列表，用于保存所有车牌的处理流程图
for plate_path in plate_paths:  # 循环处理车牌文件夹中的每一张图片
    imgTestPlate = cv_imread_chinese(plate_path)  # 使用支持中文路径的函数读取当前车牌图片
    if imgTestPlate is None:  # 判断当前图片是否读取失败
        print(f"读取失败，已跳过：{plate_path}")  # 输出读取失败提示
        continue  # 跳过读取失败的图片
    plate_stack = process_single_plate(imgTestPlate, plate_path.stem)  # 对当前车牌图片进行检测、校正、裁剪和二值化
    all_detect_results.append(plate_stack)  # 将当前处理流程图加入总列表
    cv2.imshow(f"plate_{plate_path.stem}", plate_stack)  # 显示当前车牌处理结果
    cv2.waitKey(0)  # 等待键盘输入后继续处理下一张图片
    cv2.destroyAllWindows()  # 关闭当前OpenCV显示窗口
if len(all_detect_results) > 0:  # 判断是否至少成功处理了一张车牌图像
    small_results = []  # 创建列表，用于保存缩放后的总览图
    for i, img_show in enumerate(all_detect_results):  # 遍历每一张车牌处理流程图
        small = cv2.resize(img_show, (900, 300))  # 将每张流程图统一缩放，便于总览显示
        cv2.putText(small, f"Plate {i + 1}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (238, 130, 238), 3)  # 在总览图中添加车牌序号
        small_results.append(small)  # 将缩放后的流程图加入总览列表
    overview = np.vstack(small_results)  # 将所有处理结果纵向拼接成总览图
    cv2.imwrite(str(plate_result_dir / "all_plate_process_overview.jpg"), overview)  # 保存所有车牌处理结果总览图
    cv2.imshow("all_plate_process_overview", overview)  # 显示所有车牌处理结果总览图
    cv2.waitKey(0)  # 等待键盘输入
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
# 实验结果分析：使用循环读取license plate文件夹，可以一次性处理plate1、plate2等多张车牌图像，避免一张图写一个路径
# 实验结果分析：HSV蓝色分割可以从自然彩色图像中提取蓝色车牌候选区域
# 实验结果分析：轮廓筛选可以根据面积、长宽比和填充比例定位车牌区域
# 实验结果分析：透视变换矩阵M可以将倾斜车牌拉伸校正为440×140的标准矩形图像
# 实验结果分析：车牌裁剪和字符二值化可以增强车牌字符与背景的对比度，便于后续识别或展示
######################################################################
