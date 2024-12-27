import cv2
import glob
import numpy as np
from recognition import  recognition

object = 'almond'


im_paths = glob.glob(f'../object_images/*{object}*') + glob.glob(f'../object_images/*all*') + ['../object_images/detection_tryal.png']
# im_paths =  glob.glob(f'../object_images/*all*') + ['../object_images/detection_tryal.png']

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
        # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    else:
        input('Not applying histogram equalization')

    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((12,12),np.uint8)) #battery
    cv2.imshow('After opening', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    equ_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    recognition(f'../template_images/{object}_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.5, contour_error=30)

    # recognition(f'../template_images/battery_template.png', target=equ_rgb, target_thresh=50, match_thresh=0.1, contour_error=100, template_thresh=150)

    # ALMOND
    # recognition(f'../template_images/almond_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.5, contour_error=30)
    #TAPE
    # recognition(f'../template_images/tape_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.01, contour_error=75)
    #MARKER
    # recognition(f'../template_images/marker_template.png', target=equ_rgb, target_thresh=50, match_thresh=2, contour_error=300, template_thresh=50)
    #CAP
    # recognition(f'../template_images/cap_template.png', target=equ_rgb, target_thresh=50, match_thresh=0.15, contour_error=100, template_thresh=50)
    #PIPE
    # recognition(f'../template_images/pipe_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.15, contour_error=100
    #VALVE
    # recognition(f'../template_images/valve_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.1, contour_error=50)
    # don't do opening when you don't do hist eq
    #PERFUME
    # recognition(f'../template_images/valve_template.png', target=equ_rgb, target_thresh=100, match_thresh=0.5, contour_error=150, template_thresh=70)
    # always do hist eq


    #---- TO FIX ----
    # HOOK
    #opening of 5x5 or 7x7 depending on brightness
    # don't do opening when you don't do hist eq

    # BATTERY
    #works for all images but 5ms
    #needs TEMPLATE hist eq + opening
    # template_gray = cv2.equalizeHist(template_gray)  #careful
    # template_thresh = cv2.morphologyEx(template_thresh, cv2.MORPH_OPEN, np.ones((12, 12), np.uint8)) #battery