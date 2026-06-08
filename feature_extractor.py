import cv2
import numpy as np


# Funkcja extract_features oblicza cechy z binaryzowanej maski dloni. Zwraca slownik z wizualizacjami i lista cech.
#
# Postac wywolanania:
#       vis, features = extract_features(mask)
#
# mask - binaryzowana maska, gdzie piksele dloni maja wartosc 0, a tlo ma wartosc 255. Maska powinna byc zgodna z rozmiarem obrazu.
def extract_features(image):
    # Prepare mask for processing
    mask = cv2.copyMakeBorder(
        image, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT, value=255
    )

    # Now the object is white and background black.
    mask = cv2.bitwise_not(mask)

    # Find the biggest contour.
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0]
    if len(contours) == 0:
        return {}, []

    cnt = max(contours, key=cv2.contourArea)

    # Get the convex hull and convexity defects.
    hull_points = cv2.convexHull(cnt)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)

    # Get some parameters for feature calculation.
    area_hull = cv2.contourArea(hull_points)
    area_contour = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    _, _, w, h = cv2.boundingRect(cnt)

    contour_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    defect_vis = contour_vis.copy()
    cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_vis, [hull_points], -1, (255, 0, 0), 2)
    cv2.drawContours(defect_vis, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(defect_vis, [hull_points], -1, (255, 0, 0), 2)

    defect_center_off_mass = np.zeros(2)
    depths = []

    if defects is None:
        return {}, []
    for s, e, f, d in defects[:, 0]:
        far = cnt[f][0]

        defect_center_off_mass += far
        depths.append(d / area_contour)

        start = cnt[s][0]
        end = cnt[e][0]
        cv2.line(defect_vis, start, end, (255, 0, 0), 2)
        cv2.circle(defect_vis, far, 5, (0, 0, 255), -1)

    defect_center_off_mass /= defects.shape[0]

    # Calculate the features.
    mean_dir = np.zeros(2)
    for _, _, f, _ in defects[:, 0]:
        dir = defect_center_off_mass - cnt[f][0]
        mean_dir += dir / np.linalg.norm(dir)
    mean_dir /= defects.shape[0]

    contour_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)
    black_pixels_inside_contour_prc = (
        cv2.countNonZero(cv2.bitwise_and(mask, contour_mask)) / area_contour
    )

    hull_points = cv2.convexHull(cnt)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)
    mean_dir_x, mean_dir_y = mean_dir
    mean_depth = np.mean(depths)
    max_depth = np.max(depths)
    min_depth = np.min(depths)
    std_depth = np.std(depths)
    solidity = area_contour / area_hull
    aspect_ratio = w / h
    extent = area_contour / (w * h)
    circularity = (4 * np.pi * area_contour) / (perimeter**2)

    # Organize features and visualiztion.
    features = [
        black_pixels_inside_contour_prc,
        mean_dir_x,
        mean_dir_y,
        mean_depth,
        max_depth,
        std_depth,
        solidity,
        aspect_ratio,
        extent,
        circularity,
    ]

    vis = {
        "contour": contour_vis,
        "defects": defect_vis,
    }
    return vis, features
