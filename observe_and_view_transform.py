#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 06/05/2021 23:15
# @File  : observe_and_view_transform.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
import cv2
import numpy as np
import yaml
import imutils

# Set the defalut color values for the detection result output
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x, y])


video_name = input("Enter the exact name of the video (including .mp4 or else): ")
size_frame = input("Prompt the size of the image you want to get : ")

vs = cv2.VideoCapture("C://Users//ucesxc0//Desktop//Crowded_analysis-main//video//" + video_name)
# Loop until the end of the video stream
while True:
    # Load the frame and test if it has reache the end of the video
    (frame_exists, frame) = vs.read()
    frame = imutils.resize(frame, width=int(size_frame))
    cv2.imwrite(r"C:\Users\ucesxc0\Desktop\Crowded_analysis-main\img\static_frame_from_video.jpg", frame)
    break

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)

# Load the image
img_path = r"C:\Users\ucesxc0\Desktop\Crowded_analysis-main\img\static_frame_from_video.jpg"
img = cv2.imread(img_path)

# Get the size of the image for the calibration
width, height, _ = img.shape

# Create an empty list of points for the coordinates
list_points = list()

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)

if __name__ == "__main__":
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == 4:
            pts1 = np.float32([list_points[0], list_points[1], list_points[2], list_points[3]])
            pts2 = np.float32([[0, 0], [864, 0], [0, 484], [864, 484]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst_img = cv2.warpPerspective(img, M, (width, height))
            cv2.imshow("Bird-eye view", dst_img)
            cv2.imwrite(r"C:\Users\ucesxc0\Desktop\Crowded_analysis-main\img\bird_view.jpg", dst_img)
            print('bird_view image transformation done!')

            # Return a dict to the YAML file
            config_data = dict(
                image_parameters=dict(
                    p2=list_points[3],
                    p1=list_points[2],
                    p4=list_points[0],
                    p3=list_points[1],
                    width_og=width,
                    height_og=height,
                    img_path=img_path,
                    size_frame=size_frame,
                ))
            # Write the result to the config file
            with open(r'C:\Users\ucesxc0\Desktop\Crowded_analysis-main\conf\config_birdview.yml', 'w') as outfile:
                yaml.dump(config_data, outfile, default_flow_style=False)
            break
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
