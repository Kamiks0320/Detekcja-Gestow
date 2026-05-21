import cv2
import numpy as np

FEATURE_NAMES = [
    # "defect_count",
    "mean_dir_x",
    "mean_dir_y",
    "mean_defect_depth",
    "max_defect_depth",
    # "min_defect_depth",
    "std_defect_depth",
    # "area_contour",
    # "area_hull",
    # "solidity",
    # "perimeter",
    "aspect_ratio",
    # "extent",
    # "circularity",
]


class FeatureExtractor:
    def __init__(self, image, mask_name="unknown_mask", min_contour_area_prc=0.01):
        self.image = image
        self.mask_name = mask_name
        self.min_contour_area_prc = min_contour_area_prc
        self.kernel = np.ones((3, 3), np.uint8)

    def process(self):
        gray = self._prepare_gray_image(self.image)
        binary, cnt = self._make_binary_candidate(gray)
        return self._extract_features(gray=gray, binary=binary, cnt=cnt)

    def _prepare_gray_image(self, image):
        if image is None:
            raise ValueError("Obraz wejściowy jest pusty.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.copyMakeBorder(
            gray, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT, value=255
        )

        return gray

    def _find_contours(self, binary_img):
        found = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = found[0] if len(found) == 2 else found[1]
        return contours

    def _make_binary_candidate(self, gray):
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = [th, cv2.bitwise_not(th)]

        scored = []
        img_area = gray.shape[0] * gray.shape[1]

        for bin_img in candidates:
            bin_img = cv2.morphologyEx(
                bin_img, cv2.MORPH_OPEN, self.kernel, iterations=1
            )

            bin_img = cv2.morphologyEx(
                bin_img, cv2.MORPH_CLOSE, self.kernel, iterations=2
            )

            contours = self._find_contours(bin_img)

            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            area_prc = cv2.contourArea(cnt) / img_area

            if area_prc < self.min_contour_area_prc or area_prc > 0.95:
                continue

            score = area_prc

            scored.append({"score": score, "binary": bin_img, "contour": cnt})

        if not scored:
            raise ValueError("Nie udało się znaleźć sensownego konturu w masce.")

        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[0]["binary"], scored[0]["contour"]

    def _extract_features(self, gray, binary, cnt):
        area_contour = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        hull_points = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull_points)

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull_indices = cv2.convexHull(approx, returnPoints=False)

        x, y, w, h = cv2.boundingRect(cnt)

        solidity = area_contour / area_hull if area_hull > 0 else 0.0
        aspect_ratio = w / h if h > 0 else 0.0
        extent = area_contour / (w * h) if w * h > 0 else 0.0

        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * np.pi * area_contour) / (perimeter**2)

        defects = None

        if hull_indices is not None and len(hull_indices) >= 4 and len(approx) >= 4:
            defects = cv2.convexityDefects(approx, hull_indices)

        contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        defect_vis = contour_vis.copy()

        cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(contour_vis, [hull_points], -1, (255, 0, 0), 2)

        cv2.drawContours(defect_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(defect_vis, [hull_points], -1, (255, 0, 0), 2)

        defect_list = []

        defect_points = []
        defect_center_off_mass = np.array([0, 0], dtype=np.float32)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]

                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                defect_center_off_mass += far
                defect_points.append(far)

                depth_px = d / area_contour

                cv2.line(defect_vis, start, end, (255, 0, 0), 2)
                cv2.circle(defect_vis, far, 5, (0, 0, 255), -1)

                defect_list.append(
                    {"start": start, "end": end, "far": far, "depth_px": depth_px}
                )

        defect_center_off_mass /= len(defect_points)
        mean_dir = np.array([0, 0], dtype=np.float32)
        for point in defect_points:
            dir = defect_center_off_mass - point
            x, y = dir
            l = (x * x + y * y) ** 0.5
            mean_dir += dir / l
        mean_dir /= len(defect_points)

        depths = [d["depth_px"] for d in defect_list]

        if len(depths) > 0:
            mean_depth = float(np.mean(depths))
            max_depth = float(np.max(depths))
            min_depth = float(np.min(depths))
            std_depth = float(np.std(depths))
        else:
            mean_depth = 0.0
            max_depth = 0.0
            min_depth = 0.0
            std_depth = 0.0
        # mean_dir /= area_contour
        mean_dir_x, mean_dir_y = mean_dir

        features = [
            # len(defect_list),
            mean_dir_x,
            mean_dir_y,
            mean_depth,
            max_depth,
            # min_depth,
            std_depth,
            # float(area_contour),
            # float(area_hull),
            float(solidity),
            # float(perimeter),
            float(aspect_ratio),
            float(extent),
            float(circularity),
        ]

        return {"contour": contour_vis, "defects": defect_vis}, features

    @staticmethod
    def features_to_vector(features):
        return [float(features[name]) for name in FEATURE_NAMES]


# mask = cv2.imread(r"masks\1_P_hgr1_id08_2.bmp", cv2.IMREAD_GRAYSCALE)
#
# extractor = FeatureExtractor(
#    image=mask,
#    mask_name="1_P_hgr1_id08_2.bmp"
# )
#
# vis_results, features = extractor.process()
#
# print(features)
#
#
# cv2.imshow("Gray", vis_results["gray"])
# cv2.imshow("Binary", vis_results["binary"])
# cv2.imshow("Contour + Hull", vis_results["contour_vis"])
# cv2.imshow("Defects", vis_results["defect_vis"])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
