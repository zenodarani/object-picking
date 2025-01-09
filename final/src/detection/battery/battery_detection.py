import cv2
import glob
import numpy as np
from recognition_battery import  recognition

path_to_ims = "../../../"


im_paths = glob.glob(f'{path_to_ims}object_images/*batter*') + glob.glob(f'{path_to_ims}object_images/*all*') + [f'{path_to_ims} object_images/detection_tryal.png']

images = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in im_paths]


# HISTOGRAM EQUALIZATION
for i,im in enumerate(images):
    cv2.imshow(im_paths[i], im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    brightness = np.mean(im)
    contrast = np.std(im)
    print(f"Brightness : {brightness}")
    print(f"Contrast: {contrast}")


    input('Applying histogram equalization')
    im = cv2.equalizeHist(im)

    cv2.imshow('After Histogram Equalization', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cv2.imshow('After opening', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    equ_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    recognition(f'{path_to_ims}template_images/battery_template.png', target=equ_rgb, target_thresh=60, match_thresh=0.1,
                contour_error=100, template_thresh=110)