import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self, image, min_contour_area=500):
        self.image = image
        self.min_contour_area = min_contour_area
        self.kernel = np.ones((3, 3), np.uint8)

    def process(self):
        gray = self.image
        gray = cv2.copyMakeBorder(
            gray, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT, value=0
        )

        binary, contours = self._make_binary_candidates(gray)

        cnt = max(contours, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)

        if area_cnt < self.min_contour_area:
            raise ValueError("Największy kontur jest zbyt mały.")

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull_points = cv2.convexHull(cnt)
        hull_indices = cv2.convexHull(approx, returnPoints=False)

        defects = None
        if len(approx) >= 4 and len(hull_indices) >= 4:
            defects = cv2.convexityDefects(approx, hull_indices)

        contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        defect_vis = contour_vis.copy()

        cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(contour_vis, [hull_points], -1, (255, 0, 0), 2)

        cv2.drawContours(defect_vis, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(defect_vis, [hull_points], -1, (255, 0, 0), 2)

        defect_list = []
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]

                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                depth_px = d / 256.0  # zgodnie z OpenCV fixpt_depth

                # Rysowanie: linia hull + punkt defect
                cv2.line(defect_vis, start, end, (255, 0, 0), 2)
                cv2.circle(defect_vis, far, 5, (0, 0, 255), -1)

                defect_list.append(
                    {"start": start, "end": end, "far": far, "depth_px": depth_px}
                )

        return {
            "gray": gray,
            "binary": binary,
            "contour_vis": contour_vis,
            "defect_vis": defect_vis,
        }, defect_list

    def _find_contours(self, binary_img):
        found = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        return contours

    def _make_binary_candidates(self, gray):
        # Dwie wersje: oryginalna binarizacja i odwrócona.
        # Wybieramy tę, w której największy kontur wygląda sensownie.
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

            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)

            if area < self.min_contour_area:
                continue

            # Kara za kontur prawie równy całemu obrazowi.
            penalty = 0
            if area > 0.95 * img_area:
                penalty = img_area

            score = area - penalty
            scored.append((score, bin_img, contours))

        if not scored:
            raise ValueError("Nie udało się znaleźć sensownego konturu w masce.")

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][2]
