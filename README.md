# Crowded_analysis
This is a tutorial of spectator behaviour analysis due to the COVID-19 crisis. The spectator behaviour analysis includes: detection spectator, tracking each spectator and giving them unique ID number, social distancing calculation, spectator movement distance, speed and density.



## Requirement

Development Environment: I strongly recommend to use the Anaconda virtual environment.

- OpenCV
- SKlearn
- Pillow
- Numpy 
- Tensorflow-gpu 1.15.2 (recommended)
- CUDA 10.0 

If your local machine does not have GPU, please use the **Tensorflow-CPU** to replace it. 

___

The detection and tracking is based on the **YOLOV3** and **Deep SORT**. 

- YOLOV3: detect objects on each of the video frames;
- Deep SORT: track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*



## Quick Start 

__1.Requirement__

```python
pip install -r requirements.txt 
```

__2. Download the YOLO v3 weights to your computer.__

 [[yolov3.weights]](https://pjreddie.com/media/files/yolov3.weights)

And place it in `./model_data/`

__3. Convert the Darknet YOLO model to a Keras model:__

``` 
$ python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
```

__4. Select the Region of Interest:__

```
$ python observe_and_view_transform.py 
```

When running the script, it will follow the guidance to input the video file name and the image size you want to get. In practice, the image size is commonly 800 ppx.  

__5. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ./video/testvideo.avi
```



## Camera calibration 

The camera calibration is the significant to output the desired results. The calibration is to find a Homography matrix to transfer the camera coordinate to world coordinate. The directory `camera_calibrate` gives the detail information to find the Homography matrix. When we get the 3x3 Homography matrix, it is easily to output the position of spectator in real world position. 

And also, the another concept `Meter per pixel (MPP)` is to identify the pixel and real meters transformation. For example, one spectator moves 180 pixels in videos. In order to get the real movement, it needs to use the `MPP` to transform the pixel distance into real world distance. I have already realised in the `./camera_calibrate/calibrate_img.py`. 

The file `./camera_calibrate.py` is to adjust the distortion of the camera. If your video has a severe distortion, you should firstly use the script to make a distortion adjustment. The intrinsic matrix and extrinsic matrix should be get firstly and then replace it to your camera parameters. 

___



## Reference

YOLOv3 :

```
@article{yolov3,
title={YOLOv3: An Incremental Improvement},
author={Redmon, Joseph and Farhadi, Ali},
journal = {arXiv},
year={2018}
}
```

Deep_SORT :

```
@inproceedings{Wojke2017simple,
title={Simple Online and Realtime Tracking with a Deep Association Metric},
author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
year={2017},
pages={3645--3649},
organization={IEEE},
doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
title={Deep Cosine Metric Learning for Person Re-identification},
author={Wojke, Nicolai and Bewley, Alex},
booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
year={2018},
pages={748--756},
organization={IEEE},
doi={10.1109/WACV.2018.00087}
}
```

