#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 06/05/2021 01:50
# @File  : calibrate_img.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
from __future__ import division, print_function, absolute_import
import math
import os
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
import yaml
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from keras import backend
import pandas as pd

# Set the default color values for the detection result output
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3

# MPP
# Carbao108:0.019
# Carbao205:0.016
# Carbao248:0.009
# FA-108:0.016
# FA-205:0.014
MPP = 0.014


def save_data_into_file(frame_no, time, track_id, centerX, centerY, actual_centerX, actual_centerY,
                        current_count, counter):
    """
    save the file to csv
    :param frame_no: current frame number
    :param time: frame to time
    :param track_id: tracked ID for each person
    :param centerX: center X in image coordination (unit: pixel)
    :param centerY: center Y in image coordination (unit: pixel)
    :param actual_centerX: real world x coordination position (unit: meter)
    :param actual_centerY: real world y coordination position (unit: meter)
    :param current_count: current frame total number of person
    :param counter: the whole number of people
    :param densigy: the density in current frame
    :return: the csv format data
    """
    save_dict = {'Frame_No': frame_no,
                 'Time': time,
                 'Track_ID': track_id,
                 'X coord (image)': centerX,
                 'Y coord (image)': centerY,
                 'X coord (real world)': actual_centerX,
                 'Y coord (real world)': actual_centerY,
                 'current_count': current_count,
                 'total count': counter}
    data = pd.DataFrame(save_dict)
    # output file
    data.to_csv("./output/" + args["input"] + ".csv", index=False)


def draw_rectangle(frame, corner_points):
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


def compute_velocity(center1, center2, fps):
    """
    Compute each spectator's movement speed between current frame and previous frame.
    :param center1: person's position of previous frame
    :param center2: person's position of current frame
    :param fps: the FPS parameter
    :return: the movement speed
    """
    d_pixels = math.sqrt(math.pow(center2[0] - center1[0], 2) + math.pow(center2[1] - center1[1], 2))
    d_meters = MPP * d_pixels
    speed = d_meters * fps * 3.6
    return speed, d_meters


def img_coord_to_world_coord(x, y, H):
    """
    transformation between image coordinate and real world coordinate
    :param x: x-coord in image
    :param y: y-coord in image
    :param H: Homography matrix
    :return: (acutal_x, actual_y) in real world coordinate
    """
    imagePoint = [x, y, 1]
    worldPoint = np.array(np.dot(H, imagePoint))
    scalar = worldPoint[2]
    xworld = worldPoint[0] / scalar
    yworld = worldPoint[1] / scalar

    # calibration unit is centimeter, it needs to transfer to meter
    xworld /= 100
    yworld /= 100
    xworld = round(xworld, 2)
    yworld = round(yworld, 2)
    return xworld, yworld


def frames_to_time(framerate, frames):
    """
    Transfer all the frames into time
    :param framerate: FPS
    :param frames: current frame number
    :return:timestamp（00:00:01:01）
    """
    return '{0:02d}:{1:02d}:{2:02d}:{3:02d}'.format(int(frames / (3600 * framerate)),
                                                    int(frames / (60 * framerate) % 60),
                                                    int(frames / framerate % 60),
                                                    int(frames % framerate))


def main(yolo, corner_points, H):
    """
    The main function for detection, tracking and counting the spectator for data analysis
    :param yolo: load the detection model
    :param corner_points: the observation area
    :param H: the transformation matrix from image to real-world position
    :param args: argument
    :return: the final output data that we want to get
    """
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3

    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture("./video/" + args["input"] + ".mp4")
    FPS = video_capture.get(5)  # get the video FPS
    total_frame = video_capture.get(7)  # total frame of input video
    print("video FPS is {}, and total frame is {}.".format(FPS, total_frame))

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter("./output/" + "detection_output_" + args["input"] + ".avi", fourcc, 15, (w, h))
        list_file = open("./output/" + "detection_" + args["input"] + ".txt", 'w')
        frame_index = 0

    fps = 0.0
    current_frame = []  # record the frame number
    time_stamp = []  # record the time
    indexIDs = []  # record the tracked ID
    centerX = []  # record the bounding box X-coord
    centerY = []  # record the bounding box Y-coord
    centerX_w = []  # record the real world coordinate
    centerY_w = []  # record the real world coordinate
    delta_Dis = []  # movement distance
    speed = []  # record the tracked ID's speed
    current_count = []  # record current frame person number
    counter = []  # record the total number of person
    density = []  # record the density

    # the ROI range for detection
    corner1, corner2, corner3, corner4 = corner_points
    x0, y0 = corner1
    x1, y1 = corner2
    x2, y2 = corner3
    x3, y3 = corner4
    x_min = min(x0, x1, x2, x3)
    x_max = max(x0, x1, x2, x3)
    y_min = min(y0, y1, y2, y3)
    y_max = max(y0, y1, y2, y3)
    x_range = x_max - x_min
    y_range = y_max - y_min

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        # The detection results are shown on each frame. Use the cv2.rectangle to draw the coordinates.
        for det in detections:
            bbox = det.to_tlbr()
            center0 = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            if x_min <= center0[0] < x_min + x_range and y_min <= center0[1] < y_min + y_range:
                new_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                cv2.rectangle(frame, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])),
                              (0, 0, 255), 1)

        # tracking algorithm is based on the above detection results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # each bounding boxes

            # if the person coord is over the observation area, ignore it.
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            if x_min <= center[0] < x_min + x_range and y_min <= center[1] < y_min + y_range:
                # record the time of each frame
                milliseconds = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                seconds = milliseconds // 1000
                milliseconds = milliseconds % 1000
                minutes = 0
                hours = 0
                if seconds >= 60:
                    minutes = seconds // 60
                    seconds = seconds % 60
                if minutes >= 60:
                    hours = minutes // 60
                    minutes = minutes % 60
                print(
                    "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))
                time_stamp.append(
                    "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))

                centerX.append(center[0])  # record the x coordinate
                centerY.append(center[1])  # record the y coordinate
                current_centerX = center[0]
                current_centerY = center[1]

                # transfer to world coordinate
                w_center_x, w_center_y = img_coord_to_world_coord(current_centerX, current_centerY, H)
                centerX_w.append(w_center_x)
                centerY_w.append(w_center_y)

                new_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                indexIDs.append(int(track.track_id))  # record the track ID
                counter.append(int(track.track_id))  # record the total tracked objects
                current_frame.append(frame_index)  # record the frame NO.

                # draw the rectangle for the tracking results
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])),
                              (color), 1)
                cv2.putText(frame, str(track.track_id), (int(new_bbox[0]), int(new_bbox[1] - 50)), 0, 5e-3 * 150,
                            (color), 1)

                # draw each ID into the bounding box
                if len(class_names) > 0:
                    class_name = class_names[0]
                    # cv2.putText(frame, str(class_names[0]), (int(new_bbox[0]), int(new_bbox[1] - 20)), 0, 5e-3 * 150,
                    #             (color), 1)

                # track_id[center]
                pts[track.track_id].append(center)
                thickness = 1
                # draw center point of each tracked ID
                # cv2.circle(frame, (center), 1, color, thickness)
                i += 1
                current_count.append(i)  # record the total number of person in current frame

                # draw motion path and calculate the speed of each tracked ID
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    s, delta_d = compute_velocity(pts[track.track_id][j - 1], pts[track.track_id][j], fps)
                    print("Frame NO. {}, track ID {}, center {}, "
                          "walking distance {:.2f} and speed {:.2f}".format(frame_index,
                                                                            track.track_id,
                                                                            center,
                                                                            delta_d,
                                                                            s))
                    cv2.putText(frame, str("{:.2f} m/s".format(s)), (int(new_bbox[0]), int(new_bbox[1] - 20)), 0,
                                5e-3 * 150, (color), 1)
                    cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness=1)
                    speed.append(s)  # record the tracked ID speed
                    delta_Dis.append(delta_d)  # record movement distance

                # current frame density of the selected area
                # actual_area = (x_range * MPP) * (y_range * MPP)
                # actual_area = 21.38
                # den = i / (actual_area)
                # print("Actual area:{:.2f} and density is:{:.3f}".format(actual_area, den))
                # den = round(den, 3)
                # density.append(den)

                # social distancing calculation
                # for k in range(0, len(indexIDs) - 1):
                #     if center[k] is None:
                #         continue
                #     social_distance_pixel = math.sqrt(
                #         math.pow(centerX[k] - centerX[k - 1], 2) + math.pow(centerY[k] - centerY[k - 1], 2))
                #     actual_social_distance = social_distance_pixel * MPP

        count = len(set(counter))  # total number of tracked ID in the whole video

        # draw information on the frame
        cv2.putText(frame, "Total Object Counter: " + str(count), (int(20), int(60)), 0, 5e-3 * 150, (0, 255, 0), 1)
        cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(40)), 0, 5e-3 * 150, (0, 255, 0), 1)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(20)), 0, 5e-3 * 150, (0, 255, 0), 1)
        # Draw the green rectangle to ROI zone
        observed_points = corner_points
        draw_rectangle(frame, observed_points)
        # cv2.imshow('wembley_level1', frame)
        # cv2.namedWindow("Wembley_level1", 0)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    # save all data into file
    # print(len(current_frame), len(time_stamp), len(indexIDs), len(centerX), len(centerY), len(centerX_w),
    #       len(centerY_w), len(current_count), len(counter), len(density))

    save_data_into_file(current_frame, time_stamp, indexIDs, centerX, centerY, centerX_w, centerY_w,
                        current_count, counter)

    # save_speed_and_distance(delta_Dis, speed)

    if len(pts[track.track_id]) != None:
        print(args["input"] + ": " + str(count) + " " + str(class_name) + ' Found')
    else:
        print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    backend.clear_session()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to input video", default="CA-108")
    ap.add_argument("-c", "--class", help="name of class", default="person")
    args = vars(ap.parse_args())

    pts = [deque(maxlen=30) for _ in range(9999)]
    warnings.filterwarnings('ignore')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # initialize a list of colors to represent each possible class label
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # load the ROI region for the top-down view
    print("[ Loading config file for the bird view transformation ] ")
    with open("./conf/" + args["input"] + ".yml", "r") as ymlfile:
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
    print("Done: [ Config file loaded]...")

    # load the Homography matrix for the coordinate transformation
    Homography_matrix = np.loadtxt("./camera_calibrate/cam2world_coordinate/" + args["input"] + ".txt",
                                   dtype=np.float32, delimiter=' ')
    print(Homography_matrix)
    print("Done: [ Transformation matrix loaded]...")
    main(YOLO(), corner_points, Homography_matrix)
