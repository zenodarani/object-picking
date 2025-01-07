import numpy as np
import cv2

def recognition(template_path, match_thresh=0.1, contour_error = 10, template_thresh = 120, target_thresh=120, target=None, target_path=None):
    if target is None:
        target = cv2.imread(target_path)

    with np.load('../intrinsics.npz') as item:
        mtx, dist, rvecs, tvecs = [item[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    h, w = target.shape[:2]
    cameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #undistorted_target = cv2.undistort(target, mtx, dist, None, cameramtx)[280:720, 270:1050]
    undistorted_target = target[270:760, 260:1060]

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

    # cv2.imshow('Template Binary', template_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow('Target Binary', target_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    template_contour = max(template_contours, key=lambda c: c.shape[0])

    target_with_contours = undistorted_target.copy()
    cv2.drawContours(target_with_contours, target_contours, -1, (0, 0, 255), 3)


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
    #
    # cv2.imshow('Connected Components', detected_only_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detected_only_binary, connectivity=4)

    connected_areas = []
    barycenters = []
    farthest_points = []

    coords = []
    for label in range(1, num_labels):  # Skip the background label
        component_mask = (labels == label)
        y_coords, x_coords = np.where(component_mask)

        coordinates = np.column_stack((x_coords, y_coords))
        connected_areas.append(coordinates)

        # Compute barycenter
        barycenter = np.mean(coordinates, axis=0) if len(coordinates) > 0 else None
        coords.append(coordinates)
        barycenters.append(barycenter)


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


        scale = 30
        end_point1 = (int(barycenter_temp[0] + scale * eigenvectors_temp[0][0]), int(barycenter_temp[1] + scale * eigenvectors_temp[0][1]))
        end_point2 = (int(barycenter_temp[0] + scale * eigenvectors_temp[1][0]), int(barycenter_temp[1] + scale * eigenvectors_temp[1][1]))

        if sum(target_matched[end_point2[1], end_point2[0]]) != 0:
               op = lambda a,b: a-b
        else:
            op = lambda a,b:a+b


        new_barycenter = end_point1

        distances = [np.linalg.norm(x - new_barycenter) for x in coords[i]]
        farthest_point = coords[i][np.argmax(distances)]


        angle_radians = 40

        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])

        # Rotate the eigenvectors
        rotated_eigenvector1 = np.dot(rotation_matrix, eigenvectors_temp[0])
        rotated_eigenvector2 = np.dot(rotation_matrix, eigenvectors_temp[1])

        if new_barycenter[0] < farthest_point[0]:
            additional_radians = 90
            rotation_matrix = np.array([
                [np.cos(additional_radians), -np.sin(additional_radians)],
                [np.sin(additional_radians), np.cos(additional_radians)]])

            rotated_eigenvector1 = np.dot(rotation_matrix, rotated_eigenvector1)
            rotated_eigenvector2 = np.dot(rotation_matrix, rotated_eigenvector2)


        new_end_point1 = (int(new_barycenter[0] + (scale + 5) * rotated_eigenvector1[0]),
                          int(new_barycenter[1] + (scale + 5) * rotated_eigenvector1[1]))

        new_end_point2 = (int(op(new_barycenter[0], scale * rotated_eigenvector2[0])),
                          int(op(new_barycenter[1], scale * rotated_eigenvector2[1])))

        cv2.circle(target_matched, new_barycenter, 5, (0, 0, 255), -1)

        cv2.line(target_matched, new_barycenter, new_end_point1, (255, 255, 0), 2)
        cv2.line(target_matched, new_barycenter, new_end_point2, (255, 0, 0), 2)

    cv2.imshow('Barycenter and Principal Components', target_matched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return valid_matches, means, eigenvectors