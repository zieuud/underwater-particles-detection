# underwater-particles-detection
## class_identi.py 
对5秒视频进行处理，使用了`progressive_scan_identi()`函数进行识别，将每一个检测到的颗粒物的轮廓框视为对象，以帧间轮廓框交并比为权值的最大匹配及进行目标追踪，并绘制运动轨迹线。
## match_try.py
class_identi.py 的辅助文件，存储了一些函数
> 因为参考了其他代码，所以有些乱。
