import time

import numpy as np
import pickle
from vrobot_control import vrobot
from numpy.linalg import  inv
from coordinates_transform import CoordinatesTransform
import matplotlib.pyplot as plt
import cv2
from detection.recognition import recognition

rb = vrobot('192.168.1.11')

tr = CoordinatesTransform('intrinsics.npz', 'cam2rob.pkl')

rb.move_to_pose(tr.DEFAULT_POSE, speed=600)

img = rb.grab_image()

img = tr.undistort(img)

contours, means, eigenvectors = recognition('../template_images/tape_template.png', match_thresh=0.01, contour_error=120, target=img)
moving_z = -50


scotch_offest = -16

for i in range(len(means)):
    home_speed = 600
    speed = 50
    rb.move_to_pose(tr.DEFAULT_POSE, speed=home_speed)
    rb.gripper_open()

    rb_p = tr.to_rb(means[i][0])

    pose = [rb_p[0][0], rb_p[1][0] + scotch_offest, moving_z, 179, 0, 0]
    rb.move_to_pose(pose, speed=home_speed)

    pose = [rb_p[0][0], rb_p[1][0] + scotch_offest, rb_p[2][0] + 40, 179, 0, 0]
    rb.move_to_pose(pose, speed=home_speed)

    pose = [rb_p[0][0], rb_p[1][0] + scotch_offest, rb_p[2][0] + 11, 179, 0, 0]
    rb.move_to_pose(pose, speed=speed)

    time.sleep(0.5)
    rb.gripper_close()
    time.sleep(0.5)

    pose = [rb_p[0][0], rb_p[1][0] + scotch_offest, moving_z, 179, 0, 0]
    rb.move_to_pose(pose, speed=home_speed)

    rb.move_to_pose(tr.DROP_POSE, speed=home_speed)
    time.sleep(0.5)
    rb.gripper_open()
    time.sleep(0.5)
    rb.gripper_close()
    time.sleep(0.5)
    rb.gripper_open()

    time.sleep(2)