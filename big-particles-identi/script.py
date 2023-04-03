import cv2
import math
import numpy as np
from scipy.interpolate import splprep, splev
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def point2point(t1, t2):
    """求两点之间的距离

    参数
    ----------
    t1：tuple
        长度为2，表示点的二维坐标
    t2:tuple
        长度为2，表示点的二维坐标

    返回值
    ----------
    float
        输入的两点之间的距离
    """
    dist = math.sqrt(pow(t1[0] - t2[0], 2) + pow(t1[1] - t2[1], 2))
    return dist


def point2line(k, b, point):
    """求点到直线的距离

    参数
    ----------
    k:float
        直线的斜率
    b:float
        直线的截距
    point:tuple
        长度为2，表示点的二维坐标

    返回值
    ----------
    float
        点到直线的距离
    """
    x, y = point
    d = abs(k * x - y + b) / (math.sqrt(1 + k ** 2))
    return d


def line_with_2point(point1, point2):
    """两点式求直线方程

    参数
    ----------
    point1:tuple
        长度为2，表示点的二维坐标
    point2:tuple
        长度为2，表示点的二维坐标

    返回值
    ----------
    ks:float
        直线的斜率
    bs:float
        直线的截距
    """
    x1, y1 = point1
    x2, y2 = point2
    ks = (y1 - y2) / (x1 - x2)
    bs = -(x2 * y1 - x1 * y2) / (x1 - x2)
    return ks, bs


def bisection(point1, point2, pointm):
    """求三角形内角平分线

    通过遍历顶点所对边的所有点的方式求到另外两边距离近似相等的点，该点与顶点的连线即为角平分线

    参数
    ----------
    point1:tuple
        长度为2，表示非待求顶点的二维坐标
    point2:tuple
        长度为2，表示非待求顶点的二维坐标
    pointm:tuple
        长度为2，表示待求顶点的二位坐标

    返回值
    ----------
    k:float
        角平分线的斜率
    b:float
        角平分线的截距
    """
    k1, b1 = line_with_2point(point1, pointm)
    k2, b2 = line_with_2point(point2, pointm)
    ks, bs = line_with_2point(point1, point2)
    for xi in range(min(point1[0], point2[0]), max(point1[0], point2[0])):
        yi = ks * xi + bs
        d1 = point2line(k1, b1, (xi, yi))
        d2 = point2line(k2, b2, (xi, yi))
        if abs(d1 - d2) < 10:
            k, b = line_with_2point(pointm, (xi, yi))
            return k, b
    raise ValueError("距离均不小于给定值")


def cont_fit(contour):  # 以列表的形式传入残缺的轮廓
    """拟合轮廓

    将残缺的轮廓通过B样条插值法进行拟合

    参数
    ----------
    contour:numpy.array
        数组中的每一个元素为tuple，表示轮廓点的坐标

    返回值
    ----------
    tuple
        轮廓的可绘制形式
    """
    tck, u = splprep(contour.T, u=None, s=0.5, per=1)
    u_new = np.linspace(u.min(), u.max(), 100)
    x_new, y_new = splev(u_new, tck, der=0)
    contour_new = []
    for k, j in zip(x_new, y_new):
        k = int(k)
        j = int(j)
        ll = [[k, j]]
        contour_new.append(ll)
    contour_new = np.array(contour_new)
    contour_new = (contour_new,)
    return contour_new


def targeted_hull_identi(contour):
    """判断轮廓是否有凸缺陷

    先用凸性检测筛选，再设立阈值判断是否有足够大的凸缺陷

    参数
    ----------
    contour:tuple
        通过cv2.findContours()函数的返回值

    返回值
    ----------
    list
        列表中的每一个元素为tuple，表示符合条件的凸缺陷点
    """
    if cv2.isContourConvex(contour):  # 判断是否为凸图形，如果是跳过循环
        return None
    point = []
    hull = cv2.convexHull(contour, returnPoints=False)  # 找凸包
    defects = cv2.convexityDefects(contour, hull)  # 找凸缺陷
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        triangle_ares = point2point(start, end) * d  # 计算三角形区域面积
        if triangle_ares > 500000:  # 筛掉面积过小的凸缺陷点
            point.append([f, far, start, end])
    return point


def divide(cont):
    """分离轮廓

    通过角平分线最近原则对轮廓进行一次分割

    参数
    ----------
    cont:tuple
        通过cv2.findContours()函数的返回值

    返回值
    ----------
    tuple
        通过cv2.findContours()函数的返回值
    """
    point = targeted_hull_identi(cont)
    if not point:
        return None
    elif len(point) == 1:
        return None
    elif len(point) == 2:
        min_index = min(point[0][0], point[1][0])
        max_index = max(point[0][0], point[1][0])
        cont1 = cont[:min_index].tolist() + cont[max_index:].tolist()
        cont2 = cont[min_index:max_index]
        return cont1, cont2
    else:
        for j in range(len(point)):
            k1, b1 = bisection(point[j][2], point[j][3], point[j][1])
            point_line_distance = []
            for i in range(len(point)):
                if i == j:
                    continue
                else:
                    point_line_distance.append([i, point2line(k1, b1, point[i][1])])
                point_line_distance = sorted(point_line_distance, key=lambda x: x[1])
                index2 = point_line_distance[0][0]
            k2, b2 = bisection(point[index2][2], point[index2][3], point[index2][1])
            point_line_distance = []
            for i in range(len(point)):
                if i == index2:
                    continue
                else:
                    point_line_distance.append([i, point2line(k2, b2, point[i][1])])
                point_line_distance = sorted(point_line_distance, key=lambda x: x[1])
                index1 = point_line_distance[0][0]
            if index1 == j:
                cont1 = cont[:point[index1][0]].tolist() + cont[point[index2][0]:].tolist()
                cont2 = cont[point[index1][0]:point[index2][0]]
                return cont1, cont2
            else:
                continue
        return None


def fit2cont(list_fit):
    """轮廓的可拟合形式转为可绘制形式

    参数
    ----------
    list_fit:list
        列表中的每一个元素为tuple，表示轮廓点的二维坐标

    返回值
    ----------
    tuple
        通过cv2.findContours()函数的返回值
    """
    list_new = []
    for i in list_fit:
        list_new.append([i])
    contour = (np.array(list_new),)
    return contour


# 轮廓的可绘制形式转可拟合形式
def cont2fit(contour):
    """轮廓的可绘制形式转为可拟合形式

    参数
    ----------
    contour:tuple
        通过cv2.findContours()函数的返回值

    返回值
    ----------
    list
        列表中的每一个元素为tuple，表示轮廓点的坐标
    """
    list_new = []
    for i in contour[0]:
        list_new.append(i[0])
    list_new = np.array(list_new)
    return list_new


# 计算几何特征
def geometric_features(contour):
    """计算轮廓的几何特征

    参数
    ----------
    contour:tuple
        通过cv2.findContours()函数的返回值

    返回值
    ----------
    ares:float
        轮廓的面积
    peri:float
        轮廓的周长
    roundness:float
        轮廓的圆形度
    compactness:flaot
        轮廓的密实度
    (cx,cy):tuple
        轮廓中心的坐标
    aver_radius:float
        轮廓的等效半径
    """
    ares = cv2.contourArea(contour)  # 面积
    peri = cv2.arcLength(contour, True)  # 周长
    roundness = ares * 4 * math.pi / peri ** 2  # 圆形度
    (x, y), radius = cv2.minEnclosingCircle(contour)  # 找外接圆
    compactness = ares / (math.pi * int(radius) ** 2)  # 算密实度
    m = cv2.moments(contour)  # 矩
    cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])  # 重心坐标
    distance = 0
    count = 0
    for i in contour:
        distance += point2point((cx, cy), i[0])
        count += 1
    aver_radius = distance / count
    return ares, peri, roundness, compactness, (cx, cy), aver_radius
