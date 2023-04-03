import cv2
import match_try as m
from match_try import Box
import time

time_start = time.time()

video = cv2.VideoCapture('video_sample/sample_cut.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\DeskTop\Desktop\output2.5.mp4', fourcc, 15.0, (1920, 822))

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
        dst = m.progressive_scan_identi(gray)
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect
            x -= 5 if x > 5 else x
            y -= 5 if y > 5 else y
            w += 10
            h += 10
            box = [x, y, w, h]
            if w > 300:
                continue
            if frame_count == 1:
                B = Box(box)
                target.append(B)
                locate_box_last.append(B.location)
            else:
                locate_box_now.append(box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
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
            if target[i].terminate_count < 2:
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
