import cv2
import numpy as np

def extract_features(image):
    # Prepare mask for processing
    mask = cv2.copyMakeBorder(
        image, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT, value=255
    )

    # Now the object is white and background black.
    mask = cv2.bitwise_not(mask)

    # Find contours.
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0]

    # Zabezpieczenie: brak konturów
    if len(contours) == 0:
        vis = {
            "contour": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "defects": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        }

        features = [
            0.0,  # black_pixels_inside_contour_prc * 5
            0.0,  # mean_dir_x
            0.0,  # mean_dir_y
            0.0,  # mean_depth
            0.0,  # max_depth
            0.0,  # std_depth
            0.0,  # solidity
            0.0,  # aspect_ratio
            0.0,  # extent
            0.0,  # circularity
        ]

        return vis, features

    # Find the biggest contour.
    cnt = max(contours, key=cv2.contourArea)

    # Get parameters.
    area_contour = cv2.contourArea(cnt)

    # Zabezpieczenie: kontur o zerowym polu
    if area_contour <= 0:
        vis = {
            "contour": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "defects": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        }

        features = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        return vis, features

    hull_points = cv2.convexHull(cnt)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)

    area_hull = cv2.contourArea(hull_points)
    perimeter = cv2.arcLength(cnt, True)
    _, _, w, h = cv2.boundingRect(cnt)

    # Visualizations
    contour_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    defect_vis = contour_vis.copy()

    cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_vis, [hull_points], -1, (255, 0, 0), 2)

    cv2.drawContours(defect_vis, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(defect_vis, [hull_points], -1, (255, 0, 0), 2)

    # Convexity defects
    defects = None

    if hull_indices is not None and len(hull_indices) >= 3 and len(cnt) >= 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_indices)
        except cv2.error:
            defects = None

    defect_center_off_mass = np.zeros(2, dtype=np.float64)
    depths = []

    if defects is not None and defects.shape[0] > 0:
        for s, e, f, d in defects[:, 0]:
            far = cnt[f][0].astype(np.float64)

            defect_center_off_mass += far
            depths.append(d / area_contour)

            start = cnt[s][0]
            end = cnt[e][0]

            cv2.line(defect_vis, start, end, (255, 0, 0), 2)
            cv2.circle(defect_vis, tuple(far.astype(int)), 5, (0, 0, 255), -1)

        defect_center_off_mass /= defects.shape[0]
    else:
        defect_center_off_mass = np.zeros(2, dtype=np.float64)

    # Mean direction
    mean_dir = np.zeros(2, dtype=np.float64)

    if defects is not None and defects.shape[0] > 0:
        valid_dirs = 0

        for _, _, f, _ in defects[:, 0]:
            far = cnt[f][0].astype(np.float64)
            direction = defect_center_off_mass - far
            norm = np.linalg.norm(direction)

            # Zabezpieczenie przed dzieleniem przez 0
            if norm > 0:
                mean_dir += direction / norm
                valid_dirs += 1

        if valid_dirs > 0:
            mean_dir /= valid_dirs

    mean_dir_x, mean_dir_y = mean_dir

    # Black pixels inside contour
    contour_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)

    black_pixels_inside_contour_prc = (
        cv2.countNonZero(cv2.bitwise_and(mask, contour_mask)) / area_contour
    )

    # Depth features
    if len(depths) > 0:
        mean_depth = float(np.mean(depths))
        max_depth = float(np.max(depths))
        std_depth = float(np.std(depths))
    else:
        mean_depth = 0.0
        max_depth = 0.0
        std_depth = 0.0

    # Shape features with zero division protection
    if area_hull > 0:
        solidity = area_contour / area_hull
    else:
        solidity = 0.0

    if h > 0:
        aspect_ratio = w / h
    else:
        aspect_ratio = 0.0

    if w * h > 0:
        extent = area_contour / (w * h)
    else:
        extent = 0.0

    if perimeter > 0:
        circularity = (4 * np.pi * area_contour) / (perimeter ** 2)
    else:
        circularity = 0.0

    features = [
        float(black_pixels_inside_contour_prc * 5),
        float(mean_dir_x),
        float(mean_dir_y),
        float(mean_depth),
        float(max_depth),
        float(std_depth),
        float(solidity),
        float(aspect_ratio),
        float(extent),
        float(circularity),
    ]

    vis = {
        "contour": contour_vis,
        "defects": defect_vis,
    }

    return vis, features