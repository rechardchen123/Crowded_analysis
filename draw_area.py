#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 06/05/2021 01:50
# @File  : calibrate_img.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import yaml
import cv2

# color reference
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


def draw_rectangle(frame, corner_points):
    # Draw rectangle box over the ROI area
    """
    Draw the rectangle box on each frame to indicate the observation area
    :param frame: frame from video
    :param corner_points: the observation four points
    :return: the observation area in video
    """
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]),
             COLOR_BLUE, thickness=1)
    cv2.putText(frame, str("p1"), (corner_points[0][0], corner_points[0][1]), 0, 5e-3 * 150,
                COLOR_RED, 1)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]),
             COLOR_BLUE, thickness=1)
    cv2.putText(frame, str("p2"), (corner_points[1][0], corner_points[1][1]), 0, 5e-3 * 150,
                COLOR_RED, 1)
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]),
             COLOR_BLUE, thickness=1)
    cv2.putText(frame, str("p3"), (corner_points[2][0], corner_points[2][1]), 0, 5e-3 * 150,
                COLOR_RED, 1)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]),
             COLOR_BLUE, thickness=1)
    cv2.putText(frame, str("p4"), (corner_points[3][0], corner_points[3][1]), 0, 5e-3 * 150,
                COLOR_RED, 1)


if __name__ == '__main__':
    with open("./conf/config_birdview.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    width_og, height_og = 0, 0
    corner_points = []
    for section in cfg:
        corner_points.append(cfg["image_parameters"]["p1"])
        corner_points.append(cfg["image_parameters"]["p2"])
        corner_points.append(cfg["image_parameters"]["p3"])
        corner_points.append(cfg["image_parameters"]["p4"])
        width_og = int(cfg["image_parameters"]["width_og"])
        height_og = int(cfg["image_parameters"]["height_og"])
        img_path = cfg["image_parameters"]["img_path"]
        size_frame = cfg["image_parameters"]["size_frame"]
    print(corner_points)
    print("Done: [ Config file loaded]...")

    vs = cv2.VideoCapture("./video/FA_Semi_finals/H-108.mp4")
    while True:
        # Load the frame and test if it has reache the end of the video
        (frame_exists, frame) = vs.read()
        draw_rectangle(frame, corner_points)
        cv2.imwrite("./img/FA_H-108.jpg", frame)
        break
