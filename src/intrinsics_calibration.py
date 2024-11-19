import numpy as np
import cv2
import glob

#### Calibrate intrinsic parameters:
# - Intrinsic matrix
# - (Quadratic) Distortion correction parameters
####

# We will use a chessboard as a calibrator. Do we see well al the squares?
chessboard_img = cv2.imread('../intrinsics_images/cal_1.png')
cv2.imshow('chessboard', chessboard_img)
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

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space, expressed in the calibrator's natural reference frame.
imgpoints = []  # 2d points in image plane.

# We could use multiple pictures of the calibrator, taken in different poses. In this case, we use one.
images = glob.glob("../intrinsics_images/*.png")
# images = ['CHESSBOARD.PNG']

# termination criteria for subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Find chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(chessboard_points)  # defined by hand, by us

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # associated points in the image, with subpixel accuracy

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (columns, rows), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Retrieve calibration parameters - intrinsics and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Let's have a look:
mtx  # estimated intrinsics matrix
dist  # estimated distortion coefficients k1, k2, p1, p2, k3
# p1, p2 -> k4, k5 in our slides! they are dedicated to tangential distortion

# Refine camera matrix, basing on distortion correction
img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# There is a small problem with the undistorted image
cv2.imshow('undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# See?

# Crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imshow('Undistorted and cropped', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('calibresult.png', dst)

# We can save all relevant data for further re-use
np.savez('intrinsics.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

