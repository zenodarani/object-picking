from datetime import datetime
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('QtAgg')

def date_string():
    today = datetime.now()

    # Ottieni l'anno senza il primo carattere (es: 2024 diventa 24)
    year = str(today.year)[2:]

    # Ottieni il mese in formato abbreviato (es: "Oct" diventa "OCT")
    month = today.strftime("%b").upper()

    # Ottieni il giorno con due cifre (es: 15)
    day = today.strftime("%d")

    # Componi la stringa nel formato richiesto
    return f"Y{year}{month}{day}"


def rigid_registration(model, acq):

    # Input: pnt_model [nx3], pnt_acq [nx3],
    # Output: rotation matrix R [3x3], translation vector [3x1], scale factor s

    model_bar = list(np.mean(model, 0))
    model_dev = model - model_bar

    acq_bar = list(np.mean(acq, 0))
    acq_dev = acq - acq_bar

    # mm = np.matmul(acq_dev.T,model_dev)

    scale = np.sqrt(np.sum(np.power(acq_dev, 2)) / np.sum(np.power(model_dev, 2)))
    m = acq_dev.T @ model_dev

    U, S, Vh = np.linalg.svd(m, full_matrices=True)

    R = U @ [[1, 0, 0], [0, 1, 0], [0, 0, np.sign(np.linalg.det(U @ Vh))]] @ Vh
    T = acq_bar - scale * R @ model_bar

    return R, T, scale

def build_transform_matrix(rotation: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    Given as inputs rotation_matrix [3x3] or rotation_vector [3,1] or [1,3]
    builds a 4x4 and the translation vector, builds the 4x4 transform matrix

    :param rotation: rotation component of the transform
    :param translation_vector: translation component of the transform
    :return: transform matrix
    """

    if rotation.shape == (1, 3) or rotation.shape == (3, 1):
        rotation, _ = cv2.Rodrigues(rotation)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.reshape(translation_vector, (3,))

    return matrix


def draw_frame(img, imgpts):
    origin = tuple(np.round(imgpts[0].ravel()).astype(int))
    ptx = tuple(np.round(imgpts[1].ravel()).astype(int))
    pty = tuple(np.round(imgpts[2].ravel()).astype(int))
    ptz = tuple(np.round(imgpts[3].ravel()).astype(int))
    img = cv2.line(img, origin, ptx, (0, 0, 255), 5)
    img = cv2.line(img, origin, pty, (0, 255, 0), 5)
    img = cv2.line(img, origin, ptz, (255, 0, 0), 5)
    return img

def scatter3d(pnt):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points in the 3D space
    ax.scatter(pnt[:,0], pnt[:,1], pnt[:,2], c=np.arange(0, pnt.shape[0]), cmap='viridis', marker='o', s=20)
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Show plot with rotation enabled
    plt.show(block=True)
    return fig, ax
