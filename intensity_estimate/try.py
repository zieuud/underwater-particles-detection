import cv2
import re
import pytesseract


def num_read(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dst = cv2.bilateralFilter(dst, 9, 75, 75)
    # cv2.imshow('1', dst)
    num_str = pytesseract.image_to_string(dst, config='--psm 7')
    num = re.findall(r'\d+', num_str)
    print(num)
    return None


video = cv2.VideoCapture(r"D:\desktop\college_project\video_sample.mp4")
cv2.namedWindow('1', 0)
# cv2.namedWindow('2', 0)
# cv2.namedWindow('3', 0)
# cv2.namedWindow('4', 0)
while True:
    ret, frame = video.read()
    if ret:
        roi = frame[52:707, :]
        xpos = frame[25:50, 527:640]
        ypos = frame[25:50, 656:769]
        depth = frame[25:50, 782:912]
        # data = frame[25:50, 527:912]
        cv2.imshow('1', frame)
        # cv2.imshow('2', xpos)
        # cv2.imshow('3', ypos)
        # cv2.imshow('4', depth)
        num_read(xpos)
        num_read(ypos)
        num_read(depth)
        # print(num_read(xpos))
        # print(num_read(ypos))
        # print(num_read(depth))
        cv2.waitKey(1)
    else:
        break
cv2.destroyAllWindows()
