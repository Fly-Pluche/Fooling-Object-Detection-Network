import cv2
import matplotlib.pyplot as plt

url = "rtsp://admin:tarena@10.33.68.153:8554/live"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if ret:
        # cv2.imshow("video", frame)
        # plt.imshow(frame)
        # plt.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# cap.release()
# import cv2
# import queue
# import os
# import numpy as np
# from threading import Thread
# import datetime, _thread
# import subprocess as sp
# import time
#
# # 使用线程锁，防止线程死锁
# mutex = _thread.allocate_lock()
# # 存图片的队列
# frame_queue = queue.Queue()
# # 推流的地址，前端通过这个地址拉流，主机的IP，2019是ffmpeg在nginx中设置的端口号
# rtmpUrl = "rtmp://10.33.68.153:8554/live"
# # 用于推流的配置,参数比较多，可网上查询理解
# command = ['ffmpeg',
#            '-y',
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', "{}x{}".format(640, 480),  # 图片分辨率
#            '-r', str(25.0),  # 视频帧率
#            '-i', '-',
#            '-c:v', 'libx264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'flv',
#            rtmpUrl]
#
#
# def Video():
#     # 调用相机拍图的函数
#     vid = cv2.VideoCapture(url)
#     if not vid.isOpened():
#         raise IOError("Couldn't open webcam or video")
#     while (vid.isOpened()):
#         return_value, frame = vid.read()
#
#         # 原始图片推入队列中
#         frame_queue.put(frame)
#
#
# def push_frame():
#     # 推流函数
#     accum_time = 0
#     curr_fps = 0
#     fps = "FPS: ??"
#     prev_time = time.time()
#
#     # 防止多线程时 command 未被设置
#     while True:
#         if len(command) > 0:
#             # 管道配置，其中用到管道
#             p = sp.Popen(command, stdin=sp.PIPE)
#             break
#
#     while True:
#         if frame_queue.empty() != True:
#             # 从队列中取出图片
#             frame = frame_queue.get()
#             # curr_time = time.time()
#             # exec_time = curr_time - prev_time
#             # prev_time = curr_time
#             # accum_time = accum_time + exec_time
#             # curr_fps = curr_fps + 1
#
#             # process frame
#             # 你处理图片的代码
#             # 将图片从队列中取出来做处理，然后再通过管道推送到服务器上
#             # 增加画面帧率
#             # if accum_time > 1:
#             # accum_time = accum_time - 1
#             # fps = "FPS: " + str(curr_fps)
#             # curr_fps = 0
#
#             # write to pipe
#             # 将处理后的图片通过管道推送到服务器上,image是处理后的图片
#             p.stdin.write(frame.tobytes())
#
#
# def run():
#     # 使用两个线程处理
#
#     thread1 = Thread(target=Video, )
#     thread1.start()
#     thread2 = Thread(target=push_frame, )
#     thread2.start()
#
#
# if __name__ == '__main__':
#     run()
