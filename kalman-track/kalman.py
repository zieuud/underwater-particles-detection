import numpy as np


def box2kbox(box):
    x, y, w, h = box
    kbox = [x, y, x + w, y + h]
    return kbox


def kbox2box(kbox):
    x, y, x0, y0 = kbox
    box = [x, y, x0 - x, y0 - y]
    return box


class KalmanBox:
    def __init__(self, box):
        self.A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])  # 状态转移矩阵
        self.B = None  # 控制矩阵
        self.Q = np.eye(self.A.shape[0]) * 0.1  # 过程噪声
        self.H = np.eye(self.A.shape[0])  # 观测矩阵
        self.R = np.eye(self.H.shape[0]) * 0.1  # 观测噪声
        self.box = box
        self.box_list = [box]  # 轮廓框列表
        xx, yy, ww, hh = kbox2box(box)
        self.track = [(xx + ww // 2, yy + hh // 2)]  # 轨迹存储
        self.terminate_count = 0
        self.box_predict = None  # 预测值初始化
        self.box_update = None  # 更新值初始化
        self.P_predict = np.eye(self.A.shape[0])  # 预测值协方差矩阵初始化
        self.P_update = None  # 更新值协方差矩阵初始化
        self.K = None  # 卡尔曼增益初始化
        self.Z = None  # 测量值初始化

    def predict(self):
        if len(self.box_list) == 1:
            box_predict = self.box
            box_predict.extend([0, 0, 0, 0])
            self.box_predict = box_predict
        else:
            dx1 = self.box_list[-1][0] - self.box_list[-2][0]
            dy1 = self.box_list[-1][1] - self.box_list[-2][1]
            dx2 = self.box_list[-1][2] - self.box_list[-2][2]
            dy2 = self.box_list[-1][3] - self.box_list[-2][3]
            box_predict = self.box
            box_predict.extend([dx1, dy1, dx2, dy2])
            k_box = box_predict
            self.box_predict = np.dot(self.A, np.array(k_box)).T  # 计算预测值
            self.P_predict = np.dot(np.dot(self.A, self.P_predict), self.A.T) + self.Q  # 计算预测值协方差矩阵
        return list(map(round, self.box_predict[:4]))

    def update(self, box_measure):
        dx1 = box_measure[0] - self.box[0]
        dy1 = box_measure[1] - self.box[1]
        dx2 = box_measure[2] - self.box[2]
        dy2 = box_measure[3] - self.box[3]
        box_measure.extend([dx1, dy1, dx2, dy2])
        self.Z = box_measure
        self.K = np.dot(np.dot(self.P_predict, self.H.T),
                        np.linalg.inv(np.dot(np.dot(self.H, self.P_predict), self.H.T) + self.R))  # 计算卡尔曼增益
        self.box_update = self.box_predict + np.dot(self.K,
                                                    np.array(self.Z).T - np.dot(self.H, self.box_predict))  # 计算更新值
        self.P_update = np.dot(np.eye(8) - np.dot(self.K, self.H), self.P_predict)  # 计算更新值协方差矩阵
        return list(map(round, self.box_update.tolist()[:4]))

    def done(self):
        self.box = list(map(round, self.box_update.tolist()[:4]))
        self.box_list.append(self.box)
        xx, yy, ww, hh = kbox2box(self.box)
        self.track.append((xx + ww // 2, yy + hh // 2))
        self.predict()
        return None

    def not_found(self):
        self.terminate_count += 1
        self.box = list(map(round, self.box_predict[:4]))
        self.box_list.append(self.box)
        xx, yy, ww, hh = kbox2box(self.box)
        self.track.append((xx + ww // 2, yy + hh // 2))
        return None


if __name__ == '__main__':
    box1 = [101, 101, 102, 102]  # 格式为[x, y ,w, h]
    box2 = [103, 103, 104, 104]
    box3 = [106, 106, 202, 202]
    box4 = [108, 108, 152, 152]
    KB = KalmanBox(box1)
    print('kbox:', KB.box)
    print('box2预测值：', KB.predict())
    print('box2更新值：', KB.not_found())
    KB.done()
    print('kbox:', KB.box)
    print('box3预测值：', KB.predict())
    print('box3更新值：', KB.update(box3))
    KB.done()
    print('kbox:', KB.box)
    print('box4预测值：', KB.predict())
    print('box4更新值：', KB.update(box4))
