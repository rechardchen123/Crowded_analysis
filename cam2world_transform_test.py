#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 11/05/2021 01:50
# @File  : calibrate_img.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import numpy as np
import math


def img_coord_to_world_coord(x, y, H):
    """
    transformation between image coordinate and real world coordinate
    :param x: x-coord in image
    :param y: y-coord in image
    :param H: Homography matrix
    :return: (acutal_x, actual_y) in real world coordinate
    """
    imagepoint = [x, y, 1]
    worldpoint = np.array(np.dot(H, imagepoint))
    scalar = worldpoint[2]
    xworld = worldpoint[0] / scalar
    yworld = worldpoint[1] / scalar
    xworld /= 100
    yworld /= 100
    xworld = round(xworld, 2)
    yworld = round(yworld, 2)
    return xworld, yworld


def area(x1, x2, x3, x4):
    a = x1 - x2
    d1 = math.hypot(a[0], a[1])
    b = x2 - x3
    d2 = math.hypot(b[0], b[1])
    c = x3 - x4
    d3 = math.hypot(c[0], c[1])
    d = x1 - x4
    d4 = math.hypot(d[0], d[1])
    e = x1 - x3
    d5 = math.hypot(e[0], e[1])
    print("d1, d2, d3, d4, d5:".format(d1, d2, d3, d4, d5))
    k1 = (d1 + d4 + d5) / 2
    k2 = (d2 + d3 + d5) / 2
    s1 = (k1 * (k1 - d1) * (k1 - d4) * (k1 - d5)) ** 0.5
    s2 = (k2 * (k2 - d2) * (k2 - d3) * (k2 - d5)) ** 0.5
    s = s1 + s2
    return s


if __name__ == '__main__':
    # load the Homography matrix for the coordinate transformation
    Homography_matrix = np.loadtxt("./camera_calibrate/cam2world_coordinate/Carbao_cup_coordinate_cam2world/CH-248.txt",
                                   dtype=np.float32, delimiter=' ')
    print(Homography_matrix)
    print("Done: [ Homograhy matrix loaded]...")

    # data = pd.read_csv('./output/wembley_stadium_spectator_analysis_Carbao_A-108.csv', sep=',')
    # # read the pixel coordinate, x, y
    # pixel_x = data['X coord (image)']
    # pixel_y = data['Y coord (image)']
    # world_x, world_y = img_coord_to_world_coord(pixel_x, pixel_y, Homography_matrix)

    # data.insert(data.shape[1], 'x world (new)', world_x)
    # data.insert(data.shape[1], 'y world (new)', world_y)
    # data.to_csv('./output/test.csv', index=False)

    p1_x = 370
    p1_y = 63
    p1_x1, p1_y1 = img_coord_to_world_coord(p1_x, p1_y, Homography_matrix)
    print(p1_x1, p1_y1)
    p2_x = 666
    p2_y = 167
    p2_x1, p2_y1 = img_coord_to_world_coord(p2_x, p2_y, Homography_matrix)
    print(p2_x1, p2_y1)
    p3_x = 96
    p3_y = 118
    p3_x1, p3_y1 = img_coord_to_world_coord(p3_x, p3_y, Homography_matrix)
    print(p3_x1, p3_y1)
    p4_x = 325
    p4_y = 456
    p4_x1, p4_y1 = img_coord_to_world_coord(p4_x, p4_y, Homography_matrix)
    print(p4_x1, p4_y1)

    # x1 = np.array([p1_x1, p1_y1])
    # x2 = np.array([p2_x1, p2_y1])
    # x3 = np.array([p3_x1, p3_y1])
    # x4 = np.array([p4_x1, p4_y1])

    x1 = np.array([1.51, 0.57])
    x2 = np.array([-2.11, -2.45])
    x3 = np.array([2.14, -1.46])
    x4 = np.array([-0.4, -3.32])
    a = area(x1, x2, x3, x4)
    print("area is: ".format(a))

    # x_min = min(p1_x1, p2_x1, p3_x1, p4_x1)
    # x_max = max(p1_x1, p2_x1, p3_x1, p4_x1)
    # y_min = min(p1_y1, p2_y1, p3_y1, p4_y1)
    # y_max = max(p1_y1, p2_y1, p3_y1, p4_y1)
    # print(x_min, x_max, y_min, y_max)
    #
    # x_range = x_max - x_min
    # y_range = y_max - y_min
    # print("x_range", x_range)
    # print("y_range", y_range)
    # area = x_range * y_range
    # print("area is:", area)
