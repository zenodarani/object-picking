#%%
import numpy as np
import pickle
from vrobot_control import vrobot
from numpy.linalg import  inv

rb = vrobot('192.168.1.11')

with np.load('intrinsics.npz') as item:
    K, dist, rvecs, tvecs = [item[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

tr_file = open('cam2rob.pkl', 'rb')
Tcw = pickle.load(tr_file)
tr_file.close()

#%%
P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])

px_p = [800, 550]

Zc = 924

c_xyz = inv(K) @ np.array([[Zc*px_p[0]], [Zc * px_p[1]], [Zc]])
c_xyz = np.append(c_xyz,[[1]], axis=0)

rb_p = Tcw @ c_xyz
rb_p = rb_p[:-1]

#%%
init_pos = rb.get_current_cart_pos()
rb.move_to_pose([init_pos[0], init_pos[1], init_pos[2] + 100, init_pos[3], init_pos[4], init_pos[5]], speed=600)

#%%
rb.move_to_pose([rb_p[0][0], rb_p[1][0], rb_p[2][0], 179, 1, -23], speed=600)

