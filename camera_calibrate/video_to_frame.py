#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 22:00
# @File  : video_to_frames.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import os

'''
transfer the video to images
'''
video_handle = cv2.VideoCapture('./video_for_calibrate/248.mp4')
fps = int(round(video_handle.get(cv2.CAP_PROP_FPS)))
print(fps)
frame_no = 0
save_path = './frame/'
while True:
    eof, frame = video_handle.read()
    if not eof:
        break
    frame = cv2.resize(frame, (856, 480))
    cv2.imwrite(os.path.join(save_path, "%d.jpg" % frame_no), frame)
    frame_no += 1
    cv2.waitKey(0)