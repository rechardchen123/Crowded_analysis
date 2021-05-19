#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 06/05/2021 01:50
# @File  : calibrate_img.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import matplotlib.pyplot as plt
import numpy as np

figure = plt.figure(figsize=(10, 10))
image = cv2.imread('./reference_img/Carbao-108.jpg')
# image = cv2.imread('./reference_img/Carbao-205.jpg')
# image = cv2.imread('./reference_img/Carbao-248.jpg')
# image = cv2.imread('./reference_img/FA-108.jpg')
#image = cv2.imread('./reference_img/FA-205.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# label point

# carbao-108:
square_bottom_right = np.array([352, 216])
square_bottom_left = np.array([259, 257])
square_top_left = np.array([217, 227])
square_top_right = np.array([301, 189])

# Carbao-205
# square_bottom_right = np.array([503, 413])
# square_bottom_left = np.array([454, 404])
# square_top_left = np.array([465, 368])
# square_top_right = np.array([511, 375])

# Carbao-248
# square_bottom_right = np.array([426, 449])
# square_bottom_left = np.array([373, 409])
# square_top_left = np.array([422, 369])
# square_top_right = np.array([473, 404])

# fa-108
# square_bottom_right = np.array([616, 416])
# square_bottom_left = np.array([518, 381])
# square_top_left = np.array([553, 331])
# square_top_right = np.array([639, 359])

# FA-205
# square_bottom_right = np.array([84, 353])
# square_bottom_left = np.array([33, 357])
# square_top_left = np.array([47, 316])
# square_top_right = np.array([98, 314])

plt.scatter(*square_bottom_right, color='r', s=8)
plt.scatter(*square_bottom_left, color='r', s=8)
plt.scatter(*square_top_left, color='r', s=8)
plt.scatter(*square_top_right, color='r', s=8)
rectangle = np.array([square_bottom_right, square_bottom_left, square_top_left, square_top_right, square_bottom_right])
plt.plot(rectangle[:, 0], rectangle[:, 1], color='r', ls='--')

pixel_length = np.linalg.norm(square_bottom_left - square_top_left)
# carbao-108
meters_per_pixel = 1 / pixel_length

# carbao-205
#meters_per_pixel = 0.6 / pixel_length
print('Meters/Pixel:{:.3f}'.format(meters_per_pixel))

# rectangle lengths in pixels
side = 0
side += np.linalg.norm(square_bottom_right - square_bottom_left)
side += np.linalg.norm(square_top_left - square_top_right)
side += np.linalg.norm(square_top_left - square_bottom_left)
side += np.linalg.norm(square_top_right - square_bottom_right)
side /= 4
print('Side Length: {:.2f}'.format(side * meters_per_pixel))

# Carbao-108
bottom_right = np.array([637,131])
bottom_left = np.array([152, 351])
top_left = np.array([91, 261])
top_right = np.array([512, 98])

# Carbao-205
# bottom_right = np.array([708,452])
# bottom_left = np.array([32, 330])
# top_left = np.array([205, 136])
# top_right = np.array([616, 151])

# carbao-248
# bottom_right = np.array([426,449])
# bottom_left = np.array([133, 152])
# top_left = np.array([371, 77])
# top_right = np.array([661, 189])

# fa-108
# bottom_right = np.array([667, 258])
# bottom_left = np.array([616, 415])
# top_left = np.array([159, 222])
# top_right = np.array([313, 178])

# FA-205
# bottom_right = np.array([755, 380])
# bottom_left = np.array([69, 397])
# top_left = np.array([160, 114])
# top_right = np.array([595, 86])

plt.scatter(*bottom_right, color='r', s=8)
plt.scatter(*bottom_left, color='r', s=8)
plt.scatter(*top_left, color='r', s=8)
plt.scatter(*top_right, color='r', s=8)
rectangle = np.array([bottom_right, bottom_left, top_left, top_right, bottom_right])
plt.plot(rectangle[:, 0], rectangle[:, 1], color='b', ls='--')

world_bottom_left = np.array([0, 12 * side * meters_per_pixel])
world_top_left = np.array([0, 0])
world_top_right = np.array([35 * side * meters_per_pixel, 0])
world_bottom_right = np.array([35 * side * meters_per_pixel, 12 * side * meters_per_pixel])

# Get perspective transform
image_rect = np.float32([bottom_left, top_left, top_right, bottom_right])
world_rect = np.float32([world_bottom_left, world_top_left, world_top_right, world_bottom_right])
matrix_cam2world = cv2.getPerspectiveTransform(image_rect, world_rect)
matrix_cam2world10x = cv2.getPerspectiveTransform(image_rect, world_rect * 10)
rectangle = []
for point in world_rect:
    cam_point = np.linalg.inv(matrix_cam2world) @ np.array([[point[0]], [point[1]], [1]]).reshape(3)
    rectangle.append(cam_point / cam_point[-1])
rectangle = np.array(rectangle)
print(rectangle)
print(world_rect)

# np.savetxt('./cam2world_coordinate/Carbao-108.txt', matrix_cam2world)
# np.savetxt('./cam2world_coordinate/Carbao-205.txt', matrix_cam2world)
# np.savetxt('./cam2world_coordinate/Carbao-248.txt', matrix_cam2world)
# np.savetxt('./cam2world_coordinate/FA-108.txt', matrix_cam2world)
np.savetxt('./cam2world_coordinate/FA-205.txt', matrix_cam2world)
image_rect = np.float32([bottom_left, top_left, top_right, bottom_right]).reshape(-1, 1, 2)
result = cv2.perspectiveTransform(image_rect, matrix_cam2world).squeeze()
warp = cv2.warpPerspective(image, matrix_cam2world10x, (139, 48))

plt.imshow(image)
plt.imshow(warp)
plt.show()
