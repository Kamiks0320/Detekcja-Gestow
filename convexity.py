import os
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


class HandGestureHullDefects:
    def __init__(self, image_path, min_contour_area=500):
        self.image_path = image_path
        self.min_contour_area = min_contour_area
        self.kernel = np.ones((3, 3), np.uint8)

    def load_gray(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Nie udało się wczytać obrazu: {self.image_path}")
        return img

    @staticmethod
    def _find_contours(binary_img):
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
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, self.kernel, iterations=1)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, self.kernel, iterations=2)

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

    def process(self):
        gray = self.load_gray()
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

                defect_list.append({
                    "start": start,
                    "end": end,
                    "far": far,
                    "depth_px": depth_px
                })

        return {
            "gray": gray,
            "binary": binary,
            "contour_vis": contour_vis,
            "defect_vis": defect_vis,
            "contour": cnt,
            "approx": approx,
            "hull_points": hull_points,
            "defects": defects,
            "defect_list": defect_list,
        }


def collect_mask_files(root="maski"):
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(files)


def show_results(image_path, results, save_dir=None):
    gray = results["gray"]
    binary = results["binary"]
    contour_vis = results["contour_vis"]
    defect_vis = results["defect_vis"]
    defect_count = len(results["defect_list"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Oryginał (maska)")
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Maska binarna")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Kontur + convex hull")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(defect_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Defects: {defect_count}")
    axes[3].axis("off")

    fig.suptitle(os.path.basename(image_path), fontsize=12)
    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(image_path))[0] + "_hull_defects.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Zapisano: {out_path}")


def main():
    root = "maski"
    files = collect_mask_files(root)

    if not files:
        raise FileNotFoundError(f"Nie znaleziono żadnych masek w: {root}")

    sample_n = min(4, len(files))
    random_files = random.sample(files, sample_n)

    print("Wybrane pliki:")
    for f in random_files:
        print(" -", f)

    for image_path in random_files:
        processor = HandGestureHullDefects(image_path=image_path)
        try:
            results = processor.process()
            show_results(image_path, results, save_dir="wyniki_hull_defects")
        except Exception as e:
            print(f"[BŁĄD] {image_path}: {e}")


if __name__ == "__main__":
    main()