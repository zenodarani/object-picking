import numpy as np
import pickle
from numpy.linalg import inv
import cv2


class CoordinatesTransform:

    Zc = 835

    P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])

    DEFAULT_POSE = [246, 0, -6,
                    179, 0, 0]

    DROP_POSE = [246, 0, -220,
                 179, 0, 0]

    def __init__(self, initrinsics_path, extrinsics_path):
        with np.load(initrinsics_path) as item:
            self.K, self.dist, self.rvecs, self.tvecs = [item[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

        tr_file = open(extrinsics_path, 'rb')
        self.Tcw = pickle.load(tr_file)
        tr_file.close()

    def undistort(self, img):
        h, w = img.shape[:2]
        cameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
        undistorted_target = cv2.undistort(img, self.K, self.dist, None, cameramtx)
        self.cameramtx = cameramtx
        return undistorted_target

    def to_rb(self, px_p):
        if self.cameramtx is None:
            raise Exception("undistort not called")

        c_xyz = inv(self.cameramtx) @ np.array([[CoordinatesTransform.Zc * px_p[0]],
                                      [CoordinatesTransform.Zc * px_p[1]],
                                      [CoordinatesTransform.Zc]])
        c_xyz = np.append(c_xyz, [[1]], axis=0)

        rb_p = self.Tcw @ c_xyz
        rb_p = rb_p[:-1]
        return rb_p
