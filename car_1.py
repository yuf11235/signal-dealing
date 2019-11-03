import cv2
import numpy as np
from PIL import Image
import pytesseract as tess


def dobinaryzation(img):
    """
    二值化处理函数
    """
    maxi = float(img.max())
    mini = float(img.min())

    x = maxi - ((maxi - mini) / 2)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    # 返回二值化后的黑白图像
    return thresh


# 像素拉伸
def stretch(img):
    max_ = float(img.max())
    min_ = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 255 / (max_ - min_) * img[i, j] - (255 * min_) / (max_ - min_)
    return img


def find_rectangle(contour):
    """
    寻找矩形轮廓
    """
    y, x = [], []

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def chose_licence_plate(contours, afterimg):
    """
    这个函数根据车牌的一些物理特征（面积等）对所得的矩形进行过滤
    输入：contours是一个包含多个轮廓的列表，其中列表中的每一个元素是一个N*1*2的三维数组
    输出：返回经过过滤后的轮廓集合

    拓展：
    （1） OpenCV自带的cv2.contourArea()函数可以实现计算点集（轮廓）所围区域的面积，函数声明如下：
            contourArea(contour[, oriented]) -> retval
        其中参数解释如下：
            contour代表输入点集，此点集形式是一个n*2的二维ndarray或者n*1*2的三维ndarray
            retval 表示点集（轮廓）所围区域的面积
    （2） OpenCV自带的cv2.minAreaRect()函数可以计算出点集的最小外包旋转矩形，函数声明如下：
             minAreaRect(points) -> retval
        其中参数解释如下：
            points表示输入的点集，如果使用的是Opencv 2.X,则输入点集有两种形式：一是N*2的二维ndarray，其数据类型只能为 int32
                                    或者float32， 即每一行代表一个点；二是N*1*2的三维ndarray，其数据类型只能为int32或者float32
            retval是一个由三个元素组成的元组，依次代表旋转矩形的中心点坐标、尺寸和旋转角度（根据中心坐标、尺寸和旋转角度
                                    可以确定一个旋转矩形）
    （3） OpenCV自带的cv2.boxPoints()函数可以根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点，函数声明如下：
             boxPoints(box[, points]) -> points
        其中参数解释如下：
            box是旋转矩形的三个属性值，通常用一个元组表示，如（（3.0，5.0），（8.0，4.0），-60）
            points是返回的四个顶点，所返回的四个顶点是4行2列、数据类型为float32的ndarray，每一行代表一个顶点坐标
    """
    # temp_contours = []
    # for contour in contours:
    #     if cv2.contourArea(contour) > Min_Area:
    #         temp_contours.append(contour)
    # car_plate = []
    # for temp_contour in temp_contours:
    #     rect_tupple = cv2.minAreaRect(temp_contour)
    #     rect_width, rect_height = rect_tupple[1]
    #     if rect_width < rect_height:
    #         rect_width, rect_height = rect_height, rect_width
    #     aspect_ratio = rect_width / rect_height
    #     # 车牌正常情况下宽高比在2 - 5.5之间
    #     if aspect_ratio > 2 and aspect_ratio < 4.5:
    #         car_plate.append(temp_contour)
    #         rect_vertices = cv2.boxPoints(rect_tupple)
    #         rect_vertices = np.int0(rect_vertices)
    # return car_plate
    # 
    # 找出最大的三个区域
    block = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r = find_rectangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])  # 面积
        s = (r[2] - r[0]) // (r[3] - r[1])  # 长度比

        block.append([r, a, s])
    # 选出面积最大的3个区域
    block = sorted(block, key=lambda b: b[1])[-3:]
    for rect_area in block:
        print(rect_area)

    # 使用颜色识别判断找出最像车牌的区域
    maxweight, maxindex = 0, -1
    for i in range(len(block)):
        print('block[' + str(i) + '][2]: ' + str(block[i][2]))
        if block[i][2] <=3 or block[i][2]>=5:
            continue
        b = afterimg[block[i][0][1]:block[i][0][3], block[i][0][0]:block[i][0][2]]
        # BGR转HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌的范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, lower, upper)
        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for n in w1:
            w2 += n

        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2

    return block[maxindex][0]


def license_segment(car_plate):
    """
    此函数根据得到的车牌定位，将车牌从原始图像中截取出来，并存在当前目录中。
    输入： car_plates是经过初步筛选之后的车牌轮廓的点集
    输出:   "card_img.jpg"是车牌的存储名字
    """
    global card_img
    print(len(car_plate))
    print(car_plate)
    # if len(car_plates) == 1:
    #     for car_plate in car_plates:
    # row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
    # row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
    row_min, col_min, row_max, col_max = car_plate
    cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
    card_img = img[col_min:col_max, row_min:row_max, :]
    cv2.imshow("img", img)
    cv2.imwrite("card_img.jpg", card_img)
    cv2.imshow("card_img.jpg", card_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # else:
    #     print('>>>>>>wrong!!!')
    return "card_img.jpg"


orgimg = cv2.imread('honda.jpg')
cv2.imshow('car.jpg', orgimg)
cv2.waitKey(0)

# 压缩图像
img = cv2.resize(orgimg, (400, int(400 * orgimg.shape[0] / orgimg.shape[1])))
# 灰度图
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayimg', grayimg)
cv2.waitKey(0)

stretchedimg = stretch(grayimg)
cv2.imshow('stretchedimg', stretchedimg)
cv2.waitKey(0)

# 先定义一个元素结构
r = 16
h = w = r * 2 + 1
kernel = np.zeros((h, w), dtype=np.uint8)
cv2.circle(kernel, (r, r), r, 1, -1)

# 开运算
openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)

# 获取差分图
strtimg = cv2.absdiff(stretchedimg, openingimg)

# 在对图像进行边缘检测之前，，先对图像进行二值化
binary_img = dobinaryzation(strtimg)

# 使用Canny函数做边缘检测
cannyimg = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])

# 进行闭运算
kernel = np.ones((5, 19), np.uint8)
closing_img = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)

# 进行开运算
opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)

# 再次进行开运算
kernel = np.ones((11, 5), np.uint8)
opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)

# 膨胀
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_dilated = cv2.dilate(opening_img, kernel_2)
cv2.imshow('kernel_dilated', kernel_dilated)
cv2.waitKey(0)

# eroded = cv2.erode(kernel_dilated, kernel, iterations=3)        #腐蚀图像
# cv2.imshow('kernel_dilated', eroded)
# cv2.waitKey(0)

contours, hierarchy = cv2.findContours(kernel_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# temp = np.ones(binary_img.shape,np.uint8)*255
# cv2.drawContours(temp,contours,-1,(0,255,0),3)
# cv2.imshow("contours",temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# i = 0
# for contour in contours:
#     print('This is: ' + str(i))
#     print(cv2.contourArea(contour))
#     print(contour)
#     i += 1


car_plate = chose_licence_plate(contours, img)
card_image = license_segment(car_plate)

# src = cv2.imread('card_image.jpg')
# cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input image", src)
# recognize_text()
cv2.waitKey(0)

cv2.destroyAllWindows()
