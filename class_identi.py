import cv2
import match_try as m
import time
import math

time_start = time.time()


def progressive_scan_identi(gray_input):
    for row in range(len(gray_input)):
        gray_scale = gray_input[row]
        outliers = []
        base = gray_scale[0]
        for column in range(len(gray_scale)):
            if column < len(gray_scale) - 1 and int(gray_scale[column + 1]) - int(base) > 5:
                outliers.append(column + 1)
            else:
                base = gray_scale[column + 1] if column < len(gray_scale) - 1 else 0
        for i in outliers:
            gray_input[row, i] = 255
    gray_temp = gray_input.T
    for row in range(len(gray_temp)):
        gray_scale = gray_temp[row]
        white = 0
        gap = 0
        for column in range(len(gray_scale)):
            if gray_scale[column] == 255:
                if 0 < gap < 3:
                    gray_scale[column - 1] = 255
                    gray_scale[column - 2] = 255
                white += 1
                gap = 0
            else:
                gray_scale[column] = 0
                if white == 1:
                    gray_scale[column - 1] = 0
                gap += 1
                white = 0
    gray_output = gray_temp.T
    return gray_output


class Box:
    def __init__(self, location):
        self.location = location
        self.box_track = [location]
        xx, yy, ww, hh = location
        self.track = [(xx + hh // 2, yy + ww // 2)]
        self.terminate_count = 0

    def location_add(self, new_location):
        self.location = new_location
        self.box_track.append(new_location)
        xx, yy, ww, hh = new_location
        self.track.append((xx + ww // 2, yy + hh // 2))

    def not_found(self):
        self.terminate_count += 1


def box_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    dist = math.sqrt(pow(center1[0] - center2[0], 2) + pow(center1[1] - center2[1], 2))
    return dist


def max_box(box1, box2):
    x1, y1, x11, y11 = m.mea2box(box1)
    x2, y2, x22, y22 = m.mea2box(box2)
    xi = min(x1, x2)
    yi = min(y1, y2)
    xii = max(x11, x22)
    yii = max(y11, y22)
    return m.box2mea((xi, yi, xii, yii))


def box_in_box(box1, box2):
    iou = m.cal_iou(box1, box2)
    if iou == 0:
        return box2
    else:
        x1, y1, x11, y11 = m.mea2box(box1)
        x2, y2, x22, y22 = m.mea2box(box2)
        xi = min(x1, x2)
        yi = min(y1, y2)
        xii = max(x11, x22)
        yii = max(y11, y22)
        return m.box2mea((xi, yi, xii, yii))


video = cv2.VideoCapture('video_sample/sample_cut.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\DeskTop\Desktop\output2.2.mp4', fourcc, 15.0, (1920, 822))

cv2.namedWindow('1', 0)

target = []
track_list = []
locate_box_last = []
locate_box_now = []
frame_count = 0
while True:
    ret, frame = video.read()
    if ret:
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = progressive_scan_identi(gray)
        #gauss = cv2.GaussianBlur(gray, (5, 5), 0, 30)
        #edges = cv2.Canny(gauss, 1, 30)
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect
            x -= 2 if x > 2 else x
            y -= 2 if y > 2 else y
            w += 4
            h += 4
            box = [x, y, w, h]
            if w > 300:
                continue
            if frame_count == 1:
                B = Box(box)
                target.append(B)
                locate_box_last.append(B.location)
            else:
                locate_box_now.append(box)
            x0, y0, w0, h0 = box
            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 0, 255), 1)
        if frame_count == 1:
            continue
        match_dict = m.match(locate_box_last, locate_box_now)
        # print(match_dict)
        comparison_matched_now = []
        comparison_matched_last = []
        comparison_now = list(range(0, len(locate_box_now)))
        for i in match_dict:
            index_last = int(i.split("_")[1])
            index_now = int(match_dict[i].split("_")[1])
            target[index_last].location_add(locate_box_now[index_now])
            comparison_matched_now.append(index_now)
            comparison_matched_last.append(index_last)
        unmatched_now = list(set(comparison_now) - set(comparison_matched_now))  # 找到未匹配的新目标
        unmatched_last = list(set(comparison_now) - set(comparison_matched_last))  # 找到未匹配的原目标
        for i in unmatched_now:  # 将未匹配的新目标设立为新实例
            B = Box(locate_box_now[i])
            target.append(B)
        for i in unmatched_last:  # 将未匹配的旧目标终止计数加一或删除
            if target[i].terminate_count < 1:
                target[i].not_found()
            else:
                target.pop(i)
        # print("last:", locate_box_last)
        # print("now:", locate_box_now)
        locate_box_last = []
        locate_box_now = []
        for i in target:
            locate_box_last.append(i.location)
            for j in range(len(i.track) - 1):
                cv2.line(frame, i.track[j], i.track[j + 1], (255, 0, 0), 2)
        cv2.imshow('1', frame)
        out.write(frame)
        if cv2.waitKey(5) == 's':
            break
        if frame_count == 150:
            break
    else:
        break

video.release()
out.release()
cv2.destroyAllWindows()

time_end = time.time()
time_sum = time_end - time_start
print(time_sum)
