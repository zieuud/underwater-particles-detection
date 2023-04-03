# underwater-particles-detection
## 主文件夹
> - 2023.4.3 `class_indenti.py`和`match_try.py`更新，删去非必要代码，将函数和类整合在`match_try.py`中。
### class_identi.py 
对5秒视频进行处理，使用了`progressive_scan_identi()`函数进行识别，将每一个检测到的颗粒物的轮廓框视为对象，以帧间轮廓框交并比为权值的最大匹配及进行目标追踪，并绘制运动轨迹线。
### match_try.py
class_identi.py 的辅助文件，存储了一些函数
### video sample
输入的视频
## big-particles-identi
对单帧静态大颗粒物的识别
### sample.jpg 
输入图像
### multi_overlap.py
运行的主程序，能够实现识别、重叠分离、轮廓拟合、计数、几何特征统计与分析、绘制粒径谱、提取每一个颗粒物放大图像
### script.py
辅助程序，存储一些函数
### ideal output
上述程序的理想输出
