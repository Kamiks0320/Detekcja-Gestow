import cv2
import numpy as np


FEATURE_NAMES = [
    "defect_count",
    "mean_defect_depth",
    "max_defect_depth",
    "min_defect_depth",
    "std_defect_depth",
    "area_contour",
    "area_hull",
    "solidity",
    "perimeter",
    "aspect_ratio",
    "extent",
    "circularity",
]


class FeatureExtractor:
    def __init__(
        self,
        image,
        min_contour_area=500,
        defect_min_depth=6.0,
        defect_max_angle=115.0,
    ):
        self.image = image
        self.min_contour_area = min_contour_area
        self.defect_min_depth = defect_min_depth
        self.defect_max_angle = defect_max_angle
        self.kernel = np.ones((3, 3), np.uint8)

    def process(self):
        binary = self._prepare_binary_foreground_white(self.image)

        # Ważne: czarna ramka, nie biała.
        # Biała ramka robiła kontur całego obrazu.
        binary = cv2.copyMakeBorder(
            binary,
            2,
            2,
            2,
            2,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        cnt = self._get_largest_contour(binary)

        hull_points = cv2.convexHull(cnt)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)

        defects_raw = None

        if hull_indices is not None and len(hull_indices) >= 4 and len(cnt) >= 4:
            defects_raw = cv2.convexityDefects(cnt, hull_indices)

        defect_list = []

        contour_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        defect_vis = contour_vis.copy()

        cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(contour_vis, [hull_points], -1, (255, 0, 0), 2)

        cv2.drawContours(defect_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(defect_vis, [hull_points], -1, (255, 0, 0), 2)

        if defects_raw is not None:
            for i in range(defects_raw.shape[0]):
                s, e, f, d = defects_raw[i, 0]

                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                depth_px = float(d) / 256.0
                angle_deg = self._defect_angle(start, far, end)

                # Odrzucenie mikroszumów.
                if depth_px < self.defect_min_depth:
                    continue

                # Odrzucenie zbyt szerokich wgłębień, które zwykle nie są przerwą między palcami.
                if angle_deg > self.defect_max_angle:
                    continue

                cv2.line(defect_vis, start, end, (255, 0, 0), 2)
                cv2.circle(defect_vis, far, 5, (0, 0, 255), -1)

                defect_list.append(
                    {
                        "start": start,
                        "end": end,
                        "far": far,
                        "depth_px": depth_px,
                        "angle_deg": angle_deg,
                    }
                )

        features = self._extract_numeric_features(binary, defect_list)

        vis_results = {
            "gray": binary,
            "binary": binary,
            "contour_vis": contour_vis,
            "defect_vis": defect_vis,
        }

        return vis_results, defect_list, features

    def _prepare_binary_foreground_white(self, image):
        if image is None:
            raise ValueError("Pusty obraz wejściowy.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.uint8)

        _, th = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        candidates = [
            th,
            cv2.bitwise_not(th),
        ]

        best_score = None
        best_binary = None

        img_area = gray.shape[0] * gray.shape[1]

        for candidate in candidates:
            candidate = np.where(candidate > 0, 255, 0).astype(np.uint8)

            candidate = cv2.morphologyEx(
                candidate,
                cv2.MORPH_OPEN,
                self.kernel,
                iterations=1,
            )

            candidate = cv2.morphologyEx(
                candidate,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=1,
            )

            contours = self._find_contours(candidate)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < self.min_contour_area:
                    continue

                # To usuwa przypadek, gdzie największym konturem jest całe tło/ramka.
                if area > 0.90 * img_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                touches_all_frame = (
                    x <= 1
                    and y <= 1
                    and x + w >= gray.shape[1] - 1
                    and y + h >= gray.shape[0] - 1
                )

                if touches_all_frame:
                    continue

                score = area

                if best_score is None or score > best_score:
                    clean = np.zeros_like(candidate)
                    cv2.drawContours(clean, [cnt], -1, 255, thickness=cv2.FILLED)

                    best_score = score
                    best_binary = clean

        if best_binary is None:
            raise ValueError(
                "Nie znaleziono sensownego konturu. Maska musi zawierać dłoń jako jeden główny obiekt."
            )

        return best_binary

    def _find_contours(self, binary_img):
        found = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        return found[0] if len(found) == 2 else found[1]

    def _get_largest_contour(self, binary_img):
        contours = self._find_contours(binary_img)

        if not contours:
            raise ValueError("Nie znaleziono konturu.")

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < self.min_contour_area:
            raise ValueError("Największy kontur jest zbyt mały.")

        return cnt

    def _extract_numeric_features(self, binary, defect_list):
        cnt = self._get_largest_contour(binary)

        area_contour = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)

        x, y, w, h = cv2.boundingRect(cnt)

        solidity = area_contour / area_hull if area_hull > 0 else 0.0
        aspect_ratio = w / h if h > 0 else 0.0
        extent = area_contour / (w * h) if w * h > 0 else 0.0

        circularity = 0.0

        if perimeter > 0:
            circularity = (4.0 * np.pi * area_contour) / (perimeter ** 2)

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

        features = {
            "defect_count": float(len(defect_list)),
            "mean_defect_depth": mean_depth,
            "max_defect_depth": max_depth,
            "min_defect_depth": min_depth,
            "std_defect_depth": std_depth,
            "area_contour": float(area_contour),
            "area_hull": float(area_hull),
            "solidity": float(solidity),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "extent": float(extent),
            "circularity": float(circularity),
        }

        return features

    def _defect_angle(self, start, far, end):
        a = np.array(start, dtype=np.float32)
        b = np.array(far, dtype=np.float32)
        c = np.array(end, dtype=np.float32)

        ba = a - b
        bc = c - b

        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

        if denominator == 0:
            return 180.0

        cos_value = np.dot(ba, bc) / denominator
        cos_value = np.clip(cos_value, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_value)))


def extract_features_from_mask(mask, min_contour_area=500):
    extractor = FeatureExtractor(mask, min_contour_area=min_contour_area)
    vis_results, defect_list, features = extractor.process()

    return features, vis_results, defect_list


def features_to_vector(features, feature_names=FEATURE_NAMES):
    return [float(features[name]) for name in feature_names]