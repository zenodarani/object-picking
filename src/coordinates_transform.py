import numpy as np
import pickle
from numpy.linalg import inv


class CoordinatesTransform:

    Zc = 835

    P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])

    DEFAULT_POSE = [221, -77, -6,
                    179, 0, -23]

    def __init__(self, initrinsics_path, extrinsics_path):
        with np.load(initrinsics_path) as item:
            self.K, self.dist, self.rvecs, self.tvecs = [item[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

        tr_file = open(extrinsics_path, 'rb')
        self.Tcw = pickle.load(tr_file)
        tr_file.close()

    def to_rb(self, px_p):
        c_xyz = inv(self.K) @ np.array([[CoordinatesTransform.Zc * px_p[0]],
                                        [CoordinatesTransform.Zc * px_p[1]],
                                        [CoordinatesTransform.Zc]])
        c_xyz = np.append(c_xyz, [[1]], axis=0)

        rb_p = self.Tcw @ c_xyz
        rb_p = rb_p[:-1]
        return rb_p
