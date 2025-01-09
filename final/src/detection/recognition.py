import numpy as np
import cv2

def recognition(template_path, match_thresh=0.1, contour_error = 10, template_thresh = 120, target_thresh=120, target=None, target_path=None):
    if target is None:
        target = cv2.imread(target_path)


    undistorted_target = target[270:730, 260:1060]

    template = cv2.imread(template_path)

    target_gray = cv2.cvtColor(undistorted_target, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template_gray = cv2.equalizeHist(template_gray)  #battery

    _, target_binary = cv2.threshold(target_gray, target_thresh, 255, cv2.THRESH_BINARY)
    _, template_binary = cv2.threshold(template_gray, template_thresh, 255, cv2.THRESH_BINARY)

    template_binary = cv2.morphologyEx(template_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # template_binary = cv2.morphologyEx(template_binary, cv2.MORPH_OPEN, np.ones((12, 12), np.uint8)) #battery


    target_contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contours, _ = cv2.findContours(np.uint8(template_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # cv2.imshow('Template Binary', template_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow('Target Binary', target_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    template_contour = max(template_contours, key=lambda c: c.shape[0])

    target_with_contours = undistorted_target.copy()
    cv2.drawContours(target_with_contours, target_contours, -1, (0, 0, 255), 3)

    # cv2.imshow('Template', template_thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow('Target', target_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    valid_matches = []
    template_contour_length = cv2.arcLength(template_contour, True)
    for c in target_contours:
        match = cv2.matchShapes(template_contour, c, 3, 0)
        if match <= match_thresh and abs(template_contour_length - cv2.arcLength(c,True)) <= contour_error:
            valid_matches.append(c)
        print(f"Match: {match}")
        print("Length errors: ", abs(template_contour_length - cv2.arcLength(c,True)))


    target_matched = undistorted_target.copy()
    cv2.drawContours(target_matched, valid_matches, -1, (0, 255, 0), 3)

    cv2.imshow('Target Countours', target_matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detected_only_binary = np.zeros_like(target_gray)
    cv2.drawContours(detected_only_binary, valid_matches, -1, 255, thickness=cv2.FILLED)


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detected_only_binary, connectivity=4)
    connected_areas = []
    barycenters = []

    for label in range(1, num_labels):
        component_mask = (labels == label)
        y_coords, x_coords = np.where(component_mask)

        coordinates = np.column_stack((x_coords, y_coords))
        connected_areas.append(coordinates)

        barycenters.append(np.mean(coordinates, axis=0) if len(coordinates) > 0 else None)

    means = []
    eigenvectors = []
    for i in range(len(connected_areas)):
        if len(connected_areas[i]) <= 10:
            continue

        mean_temp, eigenvectors_temp = cv2.PCACompute(connected_areas[i].astype(np.float32), mean=np.array([]))
        mean_temp[0][0] += 260
        mean_temp[0][1] += 270
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

    return valid_matches, means, eigenvectors