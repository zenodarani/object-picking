import time

import numpy as np
import pickle
from vrobot_control import vrobot
from numpy.linalg import  inv
from coordinates_transform import CoordinatesTransform
import matplotlib.pyplot as plt
import cv2
from recognition import recognition

rb = vrobot('192.168.1.11')

tr = CoordinatesTransform('intrinsics.npz', 'cam2rob.pkl')

rb.move_to_pose(tr.DEFAULT_POSE, speed=100)

img = rb.grab_image()

contours, means, eigenvectors = recognition('../template_images/almond_template.png',
                                            match_thresh=0.5, contour_error=20,
                                            target_thresh=120, template_thresh=120, target=img)

for i in range(len(means)):
    rb.move_to_pose(tr.DEFAULT_POSE, speed=100)
    rb_p = tr.to_rb(means[i][0])
    pose = [rb_p[0][0], rb_p[1][0], rb_p[2][0] + 10, 179, 0, -23]
    print(pose)

    rb.move_to_pose(pose, speed=100)
    time.sleep(3)