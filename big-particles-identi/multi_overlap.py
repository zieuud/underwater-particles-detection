import cv2
import pandas as pd
import numpy as np
import script as s
import matplotlib.pyplot as plt

pic = cv2.imread(r'C:\Users\DeskTop\Desktop\college_project/sample4.jpg')  # 读取照片
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)  # 转为灰度图
ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 二值化
dst = cv2.medianBlur(dst, 9)  # 中值模糊去噪
contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
features = []  # 创建空列表用于存储轮廓的几何特征
count = 0  # 用于计数
for cont in contours:
    ares, peri, roundness, compactness, (cx, cy), aver_radius = s.geometric_features(cont)  # 计算轮廓的各项几何特征
    if ares < 50:  # 筛掉轮廓面积过小的轮廓
        continue
    factor = (roundness + compactness) / 2  # 重叠判断因子
    if factor < 0.8:  # 判断因子小于0.8确定为重叠颗粒物
        x, y, w, h = cv2.boundingRect(cont)  # 找到外接矩形
        # 解决选取感兴趣区域(ROI)时图像边缘超出范围
        x = 3 if x < 3 else x
        y = 3 if y < 3 else y
        # 提取ROI
        pic_roi = pic[y - 3:y + h + 5, x - 3:x + w + 3]  # 原图
        pic_roi_dst = dst[y - 3:y + h + 5, x - 3:x + w + 3]  # 灰度图
        # 放大感兴趣区域
        pic_roi = cv2.resize(pic_roi, None, fx=12, fy=12, interpolation=cv2.INTER_CUBIC)  # 原图
        pic_roi_dst = cv2.resize(pic_roi_dst, None, fx=12, fy=12, interpolation=cv2.INTER_CUBIC)  # 灰度图
        pic_roi_dst = cv2.GaussianBlur(pic_roi_dst, (31, 31), 0)  # 高斯模糊使边缘平滑
        contour_roi, _ = cv2.findContours(pic_roi_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找轮廓
        for contt in contour_roi:
            # 分割轮廓
            cnt = s.divide(contt)
            if not cnt:
                ares1, peri1, roundness1, compactness1, (cx1, cy1), aver_radius1 = s.geometric_features(
                    contt)  # 计算轮廓的各项几何特征
                # 部分轮廓为单个但是被误筛为重叠的轮廓，会无法分割，在此处将其轮廓绘制并且收集几何特征
                if ares1 > 100000 or len(contour_roi) == 1:
                    cont = (cont,)  # 转为元组以用于绘制
                    count += 1  # 计数
                    features.append([count, ares, peri, roundness, compactness, (cx, cy), aver_radius])  # 将几何特征存储进列表中
                    pic = cv2.circle(pic, (cx, cy), 2, (0, 0, 0), -1)  # 画重心
                    pic = cv2.drawContours(pic, cont, -1, (255, 0, 0), 2, 1)  # 画新轮廓
                continue  # 跳过框选ROI时多余的非目标轮廓和筛选重叠颗粒物时错误选入的单个颗粒物
            else:
                cnt1, cnt2 = cnt  # divide函数返回分割剩下的两个轮廓
                cnt1, cnt2 = np.array(cnt1), np.array(cnt2)  # 转为numpy中的array数据类型
                conts = [cnt1, cnt2]  # 组成列表
                sign = True  # 指示是否跳出While循环
                while sign:
                    contss = conts  # 复制列表
                    for i in range(len(contss)):
                        cons = s.divide(contss[i])  # 对余下的轮廓再次分割，并判断是否可分
                        if cons:  # 如果可分，将分割后产生的两个轮廓替代原有轮廓，否则跳过
                            cons1, cons2 = cons
                            cons1, cons2 = np.array(cons1), np.array(cons2)
                            conts.pop(i)
                            conts.append(cons1)
                            conts.append(cons2)
                    if contss == conts:  # 如果列表中的轮廓都不可分，则复制的列表会与原列表相同，此时跳出while循环
                        sign = False
                for j in conts:  # 画出分割出来的所有轮廓
                    ares, peri, roundness, compactness, (cx, cy), aver_radius = s.geometric_features(j)  # 计算几何特征
                    ares = ares / 12 ** 2  # 将面积还原为原图大小
                    peri = peri / 12  # 将周长还原为原图大小
                    aver_radius = aver_radius / 12  # 将半径还原为原图大小
                    count += 1  # 计数
                    features.append([count, ares, peri, roundness, compactness, (cx, cy), aver_radius])
                    j = (j,)  # 转为元组
                    j = s.cont2fit(j)  # 将轮廓列表转为可拟合的形式
                    j = s.cont_fit(j)  # 拟合轮廓
                    pic_roi = cv2.circle(pic_roi, (cx, cy), 24, (0, 0, 0), -1)  # 画重心
                    pic_roi = cv2.drawContours(pic_roi, j, -1, (255, 0, 0), 24, 1)  # 画新轮廓
            pic_roi = cv2.resize(pic_roi, None, fx=(1 / 12), fy=(1 / 12), interpolation=cv2.INTER_AREA)  # 缩小ROI区域以放入原图
            pic[y - 3:y + h + 5, x - 3:x + w + 3] = pic_roi  # 将ROI区域放入原图
    else:  # 判断因子大于0.8的单个颗粒物
        cont = (cont,)  # 转为元组以用于绘制
        count += 1  # 计数
        features.append([count, ares, peri, roundness, compactness, (cx, cy), aver_radius])  # 计算几何特征
        pic = cv2.circle(pic, (cx, cy), 2, (0, 0, 0), -1)  # 画重心
        pic = cv2.drawContours(pic, cont, -1, (255, 0, 0), 2, 1)  # 画新轮廓
# 图像的保存与展示
cv2.namedWindow('1', 0)
cv2.imshow('1', pic)
cv2.imwrite(r'C:\Users\DeskTop\Desktop\final_product.jpg', pic)
cv2.waitKey()
cv2.destroyAllWindows()

# 数据分析部分
df = pd.DataFrame(features, columns=['number', 'area', 'perimeter', 'roundness', 'compactness', 'center', 'radius'],
                  dtype=float)  # 将几何特征转为DataFrame格式
# 画粒径谱
x_data = df['radius']  # 取出半径列的数据
fig, ax = plt.subplots()  # 新建子图
n, bins, pat = ax.hist(x_data, bins=20, edgecolor='r', alpha=0.75, rwidth=0.8)  # 绘制直方图
ax.plot(bins[:20], n, marker='o', color="y", linestyle="--")  # 绘制直方图上的折线图
for i in range(len(n)):
    plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")  # 打标签，在合适的位置标注每个直方图上面样本数
plt.title('Particles Size Distribution')  # 定义图表标题
plt.xlabel('radius')  # 定义x轴标签
plt.ylabel('quantity')  # 定义y轴标签
plt.savefig(r'C:\Users\DeskTop\Desktop\Particles_Size_Distribution.jpg')  # 保存图表
plt.show()  # 展示图表
# 生成几何特征数据表格
features = pd.ExcelWriter(r'C:\Users\DeskTop\Desktop\features.xlsx')
df.to_excel(features)
features.save()
