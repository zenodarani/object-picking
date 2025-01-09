import cv2
import glob
import numpy as np
from recognition import  recognition


im_paths = glob.glob(f'../../object_images/*marker*') + glob.glob(f'../../object_images/*all*') + ['../../object_images/detection_tryal.png']

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


    if brightness < 50 or contrast < 100:
        input('Applying histogram equalization')
        im = cv2.equalizeHist(im)
        cv2.imshow('After Histogram Equalization', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        input('Not applying histogram equalization')

    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cv2.imshow('After opening', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    equ_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    recognition(f'../../template_images/marker_template.png', target=equ_rgb, target_thresh=50, match_thresh=2,
                contour_error=300, template_thresh=50)