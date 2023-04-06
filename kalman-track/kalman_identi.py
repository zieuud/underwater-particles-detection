import cv2
import time
import match_try as m
from kalman import KalmanBox
import kalman as k

time_start = time.time()

video = cv2.VideoCapture('video_sample/sample_cut.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\DeskTop\Desktop\output2.8.mp4', fourcc, 15.0, (1920, 822))

cv2.namedWindow('1', 0)

target = []
locate_box_last = []
locate_box_now = []
frame_count = 0
while True:
    ret, frame = video.read()
    if ret:
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = m.progressive_scan_identi(gray)
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect
            x -= 10 if x > 10 else x
            y -= 10 if y > 10 else y
            w += 20
            h += 20
            box = [x, y, x + w, y + h]
            if w > 300:
                continue
            if frame_count == 1:
                KB = KalmanBox(box)
                target.append(KB)
                KB.predict()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)  # 绘制第一帧检测框
            else:
                locate_box_now.append(box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)  # 绘制测量值
        if frame_count == 1:
            continue
        locate_box_last = [x.box_predict[:4] for x in target]
        for predict_box in locate_box_last:
            x1_p, y1_p, x2_p, y2_p = predict_box
            cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 1)  # 绘制预测值
        # 本帧轮廓框与上一帧轮廓框匹配
        match_dict = m.match(locate_box_last, locate_box_now)
        # print(match_dict)
        comparison_matched_now = []
        comparison_matched_last = []
        comparison_now = list(range(0, len(locate_box_now)))
        for i in match_dict:
            index_last = int(i.split("_")[1])
            index_now = int(match_dict[i].split("_")[1])
            update_box = target[index_last].update(locate_box_now[index_now])  # 更新步
            target[index_last].done()  # 结束一个卡尔曼周期并归并
            comparison_matched_now.append(index_now)
            comparison_matched_last.append(index_last)
        unmatched_now = list(set(comparison_now) - set(comparison_matched_now))  # 找到未匹配的新目标
        unmatched_last = list(set(comparison_now) - set(comparison_matched_last))  # 找到未匹配的原目标
        for i in unmatched_now:  # 将未匹配的新目标设立为新实例
            KB = KalmanBox(locate_box_now[i])
            target.append(KB)
            KB.predict()
        for i in unmatched_last:  # 将未匹配的旧目标终止计数加一或删除
            if target[i].terminate_count < 2:
                target[i].not_found()
            else:
                target.pop(i)
        #for kb in target:
            #x1_u, y1_u, x2_u, y2_u = kb.box[:4]
            #cv2.rectangle(frame, (x1_u, y1_u), (x2_u, y2_u), (0, 0, 255), 1)  # 绘制更新值
        # print("last:", locate_box_last)
        # print("now:", locate_box_now)
        locate_box_last = []
        locate_box_now = []
        for i in target:
            for j in range(len(i.track) - 1):
                cv2.line(frame, i.track[j], i.track[j + 1], (255, 0, 0), 2)
        cv2.imshow('1', frame)
        out.write(frame)
        print(frame_count)
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
