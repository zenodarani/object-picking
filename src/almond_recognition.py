import numpy as np
import cv2

#%%
# target = cv2.imread('object_images/almonds_100ms.png')[250:, 200:1080]
target = cv2.imread('./template_images/detection_tryal.png')[250:750, 210: 1100]
# target = cv2.imread('object_images/all_50ms.png')
template = cv2.imread('./template_images/almond_template.png')

target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#%%
_, target_thresh = cv2.threshold(target_gray, 120, 255, cv2.THRESH_BINARY)
_, template_thresh = cv2.threshold(template_gray, 120, 255, cv2.THRESH_BINARY)

#%%
target_contours, _ = cv2.findContours(target_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

template_contour = template_contours[0]
#%%
# template_with_contours = template.copy()
# cv2.drawContours(template_with_contours, template_contours, -1, (0,0,255), 3)
# cv2.imshow("Contours", template_with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#%%
target_with_contours = target.copy()
cv2.drawContours(target_with_contours, target_contours, -1, (0,0,255), 3)
cv2.imshow("Contours", target_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
n_almonds = 12
best_distance = float('inf')
error = 20

valid_matches = []
template_contour_length = cv2.arcLength(template_contour, True)

for c in target_contours:
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    if match <= 0.5 and template_contour_length - error <= cv2.arcLength(c, True) <= template_contour_length + error:
        valid_matches.append(c)
#%%
target_matched = target.copy()
cv2.drawContours(target_matched, valid_matches, -1, (0, 255, 0), 3)
cv2.imshow('Matched shape', target_matched)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
detected_only_binary = np.zeros_like(target_gray)
cv2.drawContours(detected_only_binary, valid_matches, -1, 255, thickness=cv2.FILLED)
cv2.imshow('Matched shape', detected_only_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detected_only_binary, connectivity=4)

connected_areas = []
barycenters = []

for label in range(1, num_labels):
    component_mask = (labels == label)
    y_coords, x_coords = np.where(component_mask)

    coordinates = np.column_stack((x_coords, y_coords))
    connected_areas.append(coordinates)

    barycenters.append(np.mean(coordinates, axis=0) if len(coordinates) > 0 else None)

print(f"N components: {len(connected_areas)}")




#%%
output_image = target.copy()
means = []
eigenvectors = []
for i in range(len(connected_areas)):
    if len(connected_areas[i]) <= 10:
        continue

    mean_temp, eigenvectors_temp = cv2.PCACompute(connected_areas[i].astype(np.float32), mean=np.array([]))
    means.append(mean_temp)
    eigenvectors.append(eigenvectors_temp)

    barycenter_temp = tuple(barycenters[i].astype(int))
    cv2.circle(target_matched, barycenter_temp, 5, (0, 0, 255), -1)

    scale = 50
    for vec in eigenvectors_temp:
        end_point = (int(barycenter_temp[0] + scale * vec[0]), int(barycenter_temp[1] + scale * vec[1]))
        cv2.line(target_matched, barycenter_temp, end_point, (255, 0, 0), 2)

cv2.imshow('Barycenter and Principal Components', target_matched)
cv2.waitKey(0)
cv2.destroyAllWindows()