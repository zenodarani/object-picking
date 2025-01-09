import numpy as np
import cv2
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imagesupport as ims
from vrobot_control import vrobot

matplotlib.use('QtAgg')

#### Calibrate extrinsic matrix

# Load saved intrinsics and distortion correction parameters
with np.load('intrinsics.npz') as item:
    mtx, dist, rvecs, tvecs = [item[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Let's repeat all the last steps of the intrinsics + distortion calibration
# Undistort
img = cv2.imread('../extrinsics_images/extrinsic_cal.png')
h,  w = img.shape[:2]
cameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
img_udst = cv2.undistort(img, mtx, dist, None, cameramtx)

# Do we see the calibration plate well?
cv2.imshow('chessboard_undistorted', img_udst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# These numbers must match the used calibration grid. The grid is formed by the internal intersection of the chessboard
square_size = 34
rows = 7
columns = 9

# Coordinates of calibration grid points w.r.t. associated frame
chessboard_points = np.zeros((columns * rows, 3), np.float32)
chessboard_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
chessboard_points *= square_size

# Find the chess board corners
gray = cv2.cvtColor(img_udst, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)

# Termination criteria for subpixel accuracy corner search algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Refine corner locations
corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Compute the chessboard reference frame origin and orientation w.r.t. camera reference frame
ret, rvec, tvec = cv2.solvePnP(chessboard_points, corners_subpix, cameramtx, np.zeros(5))

# Draw and display the corners
img_temp = img_udst.copy()
cv2.drawChessboardCorners(img_temp, (columns, rows), corners_subpix, ret)
cv2.drawFrameAxes(img_temp, cameramtx, np.zeros(5), rvec, tvec, length=30.0, thickness=3)
cv2.imshow('detected corners', img_temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# List the indices of the corners you touched with the robot (at least 4!)
#ind_corner = [0, columns - 1, (rows - 1) * columns, ] # or whatever i<ndices you like, e.g. i choose:
#ind_corner = [0, columns-1, (rows-1)*columns, rows*columns -1]
ind_corner = [0, 8, 45, 53]

# Fill pnt_robt with corner coordinates w.r.t. the robot's EE. Mind the order of the corners!
#pnt_rob = np.array([[0, 0, 0],
#                    [(columns-1)*square_size, 0, 0],
#                    [0, (rows-1)*square_size, 0],
#                    [(columns-1)*square_size, (rows-1)*square_size, 0]])
#pnt_rob = chessboard_points[ind_corner]
pnt_rob = np.array([[321.325, -146.509, -303.526],
                    [320.060, 126.883, -308.561],
                    [493.484, -146.229, -307.527],
                    [492.934, 128.922, -307.423]])
n_pnt = len(ind_corner)

fig, ax = ims.scatter3d(pnt_rob)

pnt_calib = chessboard_points[ind_corner]
rmat = cv2.Rodrigues(rvec)[0]  # rvec is represented as axis-angle. Convert to rotation matrix
pnt_cam = (rmat @ pnt_calib.T + tvec).T  # Turn points into camera reference frame

# Visualize
fig, ax = ims.scatter3d(pnt_calib)

# Solve the equation " pnt_rob = s * R * pnt_cam + T " in the least squares sense
R, T, scale = ims.rigid_registration(pnt_cam, pnt_rob)
print(f"Pnts cam \n",pnt_cam)
print(f"Pnts rob \n", pnt_rob)
print(f"Estimated robot-camera transformation: R, T \n", R, T)

# Test
pnt_rob_recomputed = R @ pnt_cam.T + np.tile(T[:, np.newaxis], (1, pnt_cam.shape[0]))
# ok!

# Assemble homogeneous transform
Tr = ims.build_transform_matrix(R, T)

# Save it for later use
tr_file = open('cam2rob.pkl', 'wb')
pickle.dump(Tr, tr_file)
# close the file
tr_file.close()


## Let's try to visualize the chessboard frame, expressed in robot coordinates
pnt_frame = pnt_rob[0:3, :]
z_point = np.cross(pnt_frame[1, :] - pnt_frame[0, :], pnt_frame[2, :] - pnt_frame[0, :])/1000
pnt_frame = np.vstack((pnt_frame, z_point))

# Convert robot coordinates to image frame coordinates
pnt_frame_img = (R.T @ (pnt_frame - T).T).T
fig, ax = ims.scatter3d(pnt_frame_img)

# project 3D points to image plane
#imgpts, jac = cv2.projectPoints(pnt_frame_img, rvec, tvec, mtx, dist)
pix = (cameramtx @ pnt_frame_img.T).T / pnt_frame_img[0, 2]

img_frame = ims.draw_frame(img_udst, pix[:, 0:2])
cv2.imshow('img', img_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

