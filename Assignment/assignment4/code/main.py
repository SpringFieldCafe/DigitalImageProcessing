import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path

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
        if y%100==0:  # 每处理100行输出一次进度
            sys.stdout.write(f"双线性插值处理中：{y}/{dst_h}\n")  # 使用sys模块输出当前处理进度
            sys.stdout.flush()  # 立即刷新输出缓冲区
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
    sys.stdout.write("双线性插值处理完成\n")  # 使用sys模块输出处理完成提示
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
sys.stdout.write("第二题：正在执行最近邻插值和双线性插值自定义放大程序\n")  # 调用sys模块输出程序运行提示信息
sys.stdout.write(f"当前Python解释器路径：{sys.executable}\n")  # 调用sys模块输出当前Python解释器路径
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
sys.stdout.write("第二题：原图、最近邻插值图、双线性插值图已分别单独保存\n")  # 使用sys模块输出单独保存完成提示
sys.stdout.flush()  # 刷新标准输出缓冲区
cv2.imshow("Original Self Image 2",original_text)  # 单独使用OpenCV窗口显示原始自拍图像
cv2.waitKey(0)  # 等待键盘按键后继续显示下一张图
cv2.imshow("Nearest Neighbor x1.5",nearest_text)  # 单独使用OpenCV窗口显示最近邻插值结果图像
cv2.waitKey(0)  # 等待键盘按键后继续显示下一张图
cv2.imshow("Bilinear Interpolation x1.5",bilinear_text)  # 单独使用OpenCV窗口显示双线性插值结果图像
cv2.waitKey(0)  # 等待键盘按键
cv2.destroyAllWindows()  # 关闭所有OpenCV显示窗口
######################################################################