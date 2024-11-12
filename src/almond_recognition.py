import numpy as np
import cv2

#%%
target = cv2.imread('object_images/all_50ms.png')
template = cv2.imread('template_images/almond_template.png')

target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#%%
_, target_thresh = cv2.threshold(target_gray, 120, 255, cv2.THRESH_BINARY)
_, template_thresh = cv2.threshold(template_gray, 120, 255, cv2.THRESH_BINARY)

#%%
target_contours, _ = cv2.findContours(target_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#%%
target_with_contours = target.copy()
cv2.drawContours(target_with_contours, target_contours, -1, (0,0,255), 3)
cv2.imshow("Contours", target_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
