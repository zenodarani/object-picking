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

img = tr.undistort(img)

contours, means, eigenvectors = recognition('../template_images/battery_template.png', match_thresh=1,
                                            contour_error=66, template_thresh=230, target_thresh=120, target=img)
moving_z = -50


left_offset = -1

for i in range(len(means)):
    home_speed = 100
    speed = 50
    rb.move_to_pose(tr.DEFAULT_POSE, speed=speed)
    rb.gripper_open()

    rb_p = tr.to_rb(means[i][0])

    pose = [rb_p[0][0], rb_p[1][0], moving_z, 179, 0, np.arctan2(*eigenvectors[i][0]) * 180 / np.pi - 180]
    rb.move_to_pose(pose, speed=speed)

    pose = [rb_p[0][0], rb_p[1][0], rb_p[2][0] + 11, 179, 0, np.arctan2(*eigenvectors[i][0]) * 180 / np.pi - 180]
    rb.move_to_pose(pose, speed=speed)

    time.sleep(0.5)
    rb.gripper_close()
    time.sleep(0.5)

    pose = [rb_p[0][0], rb_p[1][0], moving_z, 179, 0, np.arctan2(*eigenvectors[i][0]) * 180 / np.pi - 180]
    rb.move_to_pose(pose, speed=speed)

    rb.move_to_pose(tr.DROP_POSE, speed=speed)
    time.sleep(0.5)
    rb.gripper_open()
    time.sleep(0.5)
    rb.gripper_close()
    time.sleep(0.5)
    rb.gripper_open()

    time.sleep(3)