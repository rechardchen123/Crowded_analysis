#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 06/05/2021 00:38
# @File  : camera_calibrate.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import numpy as np
import cv2


def projection_matrices():
    # intrinsic matrix
    F_X =   233.621663208811
    F_Y =  76.8862589911439
    C_X =  1110.37298074342
    C_Y =-140.627597875706
    intrinsic_matrix = np.array([
        [F_X, 0, C_X],
        [0, F_Y, C_Y],
        [0, 0, 1]
    ])

    # extrinsic matrix
    rotation_matrix = np.array([
        [  -0.8835   , 0.3635   , 0.2955],
        [   0.4067  ,  0.9082  ,  0.0990],
        [  -0.2324   , 0.2076  , -0.9502]])
    translation = np.array([[-553.444896747179], [ 1173.00296731762], [218.236910215249]])
    extrinsic_matrix = np.concatenate((rotation_matrix, translation), axis=1)
    return intrinsic_matrix, extrinsic_matrix


def project_w2c(p, in_mat, ex_mat, distortion=False):
    # extrinsic
    P = np.array(p).reshape(4, 1)
    p_temp = ex_mat @ P

    # distortion
    if distortion:
        K1 = 0.0213
        K2 = -0.1164
        P1 = 0
        P2 = 0
        x_p = p_temp[0][0]
        y_p = p_temp[1][0]
        r_sq = x_p ** 2 + y_p ** 2
        xpp = x_p * (1 + K1 * r_sq + K2 * (r_sq ** 2)) + 2 * P1 * x_p * y_p + P2 * (r_sq + 2 * (x_p ** 2))
        ypp = y_p * (1 + K1 * r_sq + K2 * (r_sq ** 2)) + 2 * P2 * x_p * y_p + P1 * (r_sq + 2 * (y_p ** 2))
        p_temp[0][0] = xpp
        p_temp[1][0] = ypp

    # intrinsic
    p = in_mat @ p_temp
    p = p / p[2]
    return np.int(p[0]), np.int(p[1])


def convert_background(multiplier=52):
    img = cv2.imread('./camera_calibrate/wembley_stadium_background.jpg')
    HEIGHT = img.shape[0]
    WIDHT = img.shape[1]
    h_cal = 50 * multiplier  # 52 pixel = 1 meter
    w_cal = 50 * multiplier
    img_cal = np.zeros((h_cal, w_cal, 3))
    in_mat, ex_mat = projection_matrices()
    for i in range(h_cal):
        for j in range(w_cal):
            x, y = project_w2c([i / multiplier, j / multiplier, 0, 1], in_mat, ex_mat)
            if 0 <= y < HEIGHT and 0 <= x < WIDHT:
                img_cal[i, j, :] = img[y, x, :]

    cv2.imwrite('./camera_calibrate/wembley_stadium_background_calibrated.jpg', img_cal)
    print('Calibrated background saved.')


def save_transformation_matrix():
    in_mat, ex_mat = projection_matrices()
    M_c2w = np.linalg.inv(in_mat @ np.delete(ex_mat, 2, axis=1))
    np.savetxt('./camera_calibrate/wembley_stadium_cam2world1.txt', M_c2w)
    print('Transformation matrix saved.')


if __name__ == "__main__":
    F_X = 893.2599
    F_Y = 2837.4
    C_X = 307.3224
    C_Y = 212.5938

    intrinsic_matrix = np.array([
        [F_X, 0, C_X],
        [0, F_Y, C_Y],
        [0, 0, 1]])

    rotation_matrix = np.array([
        [-0.9133, 0.1161, -0.3903],
        [-0.4008, -0.0870, 0.9120],
        [0.0719, 0.9894, 0.1260]])

    translation = np.array([[360.7214], [21.0471], [670.0632]])

    imagePoint = [259, 257, 1]
    zConst = 0
    # calculate the scale factor
    tempMat = np.linalg.inv(rotation_matrix) * np.linalg.inv(intrinsic_matrix) * imagePoint
    tempMat2 = np.linalg.inv(rotation_matrix) * translation
    print(tempMat)
    print(tempMat2)
    s = zConst + tempMat2[2, 0]
    s /= tempMat[2, 0]
    print("scale factor is: ", s)

    # camera to world transformation
    c2w = np.linalg.inv(rotation_matrix) * (np.linalg.inv(intrinsic_matrix) * s * imagePoint - translation)
    print('camera to world: ', c2w)

    # convert_background()
    save_transformation_matrix()
