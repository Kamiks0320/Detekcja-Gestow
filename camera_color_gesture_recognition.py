import json
from pathlib import Path
from pyexpat import features

import cv2
import numpy as np

from feature_extractor import FeatureExtractor


DATABASE_PATH = Path("features_database") / "features_database.txt"

CAMERA_INDEX = 0
OUTPUT_SIZE = 256

# ============================================================
# TUTAJ WYBIERASZ TRYB SEGMENTACJI:
#
# "color"      -> wybór koloru skóry kliknięciem myszą
# "background" -> odejmowanie tła
# ============================================================
#SEGMENTATION_MODE = "color"
SEGMENTATION_MODE = "background"


# tolerancje dla HSV wokół klikniętego koloru skóry
H_TOL = 12
S_TOL = 60
V_TOL = 80

# parametry odejmowania tła
BACKGROUND_FRAMES = 40
BACKGROUND_DIFF_THRESHOLD = 15
BACKGROUND_BLUR_SIZE = 7

# czyszczenie maski
KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((7, 7), np.uint8)

MIN_CONTOUR_AREA = 800
ROI_MARGIN = 25

USE_FIXED_ROI = True

ROI_X1 = 120
ROI_Y1 = 80
ROI_X2 = 520
ROI_Y2 = 420

# debug
PRINT_FEATURES = True
PRINT_EVERY_N_FRAMES = 20
def print_features(features, database, frame_counter):
    if not PRINT_FEATURES:
        return

    if frame_counter % PRINT_EVERY_N_FRAMES != 0:
        return

    print("\n===== FEATURES FROM CAMERA =====")

    for name in database["feature_names"]:
        value = features[name]
        print(f"{name:22s}: {value:.6f}")

    print("================================\n")


def load_database(path=DATABASE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono bazy cech: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_contours(binary_img):
    found = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return contours


def features_to_vector(features, feature_names):
    return [float(features[name]) for name in feature_names]


def scale_vector(vector, mean, std):
    vector = np.array(vector, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    std[std == 0] = 1.0

    return ((vector - mean) / std).tolist()


def euclidean_distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    return float(np.linalg.norm(a - b))


def predict_by_centroid(features, database):
    feature_names = database["feature_names"]

    vector = features_to_vector(features, feature_names)

    scaler_mean = database["scaler"]["mean"]
    scaler_std = database["scaler"]["std"]

    vector_scaled = scale_vector(vector, scaler_mean, scaler_std)

    distances = {}

    for label, template in database["templates"].items():
        template_vector = template["mean_vector_scaled"]

        distance = euclidean_distance(vector_scaled, template_vector)
        distances[label] = distance

    predicted_label = min(distances, key=distances.get)

    return predicted_label, distances


def extract_numeric_features(binary, defect_list):
    contours = find_contours(binary)

    if not contours:
        raise ValueError("Nie znaleziono konturu w obrazie binarnym.")

    cnt = max(contours, key=cv2.contourArea)

    area_contour = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    area_hull = cv2.contourArea(hull)

    x, y, w, h = cv2.boundingRect(cnt)

    solidity = area_contour / area_hull if area_hull > 0 else 0
    aspect_ratio = w / h if h > 0 else 0
    extent = area_contour / (w * h) if w * h > 0 else 0

    circularity = 0
    if perimeter > 0:
        circularity = (4 * np.pi * area_contour) / (perimeter ** 2)

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
        "defect_count": len(defect_list),
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


# ============================================================
# TRYB 1: SEGMENTACJA PO KOLORZE
# ============================================================

def make_skin_mask(frame_bgr, selected_hsv):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = selected_hsv

    h_low = h - H_TOL
    h_high = h + H_TOL

    s_low = max(s - S_TOL, 0)
    s_high = min(s + S_TOL, 255)

    v_low = max(v - V_TOL, 0)
    v_high = min(v + V_TOL, 255)

    if h_low < 0:
        lower1 = np.array([0, s_low, v_low], dtype=np.uint8)
        upper1 = np.array([h_high, s_high, v_high], dtype=np.uint8)

        lower2 = np.array([180 + h_low, s_low, v_low], dtype=np.uint8)
        upper2 = np.array([179, s_high, v_high], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

    elif h_high > 179:
        lower1 = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper1 = np.array([179, s_high, v_high], dtype=np.uint8)

        lower2 = np.array([0, s_low, v_low], dtype=np.uint8)
        upper2 = np.array([h_high - 180, s_high, v_high], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

    else:
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

    mask = clean_mask(mask)

    return mask


def choose_color_from_camera(cap):
    selected_hsv = {"value": None}
    selected_bgr = {"value": None}

    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Nie udało się pobrać obrazu z kamery.")

    clone = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            patch_size = 5

            y1 = max(y - patch_size, 0)
            y2 = min(y + patch_size + 1, clone.shape[0])
            x1 = max(x - patch_size, 0)
            x2 = min(x + patch_size + 1, clone.shape[1])

            patch = clone[y1:y2, x1:x2]

            mean_bgr = np.mean(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            selected_bgr["value"] = mean_bgr.tolist()

            hsv_pixel = cv2.cvtColor(
                np.uint8([[mean_bgr]]),
                cv2.COLOR_BGR2HSV
            )[0][0]

            selected_hsv["value"] = hsv_pixel.tolist()

            cv2.circle(clone, (x, y), 8, (0, 0, 255), -1)

            print("Wybrany BGR:", selected_bgr["value"])
            print("Wybrany HSV:", selected_hsv["value"])

    window_name = "Wybierz kolor skory - kliknij na dlon"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)


    while True:
        display = clone.copy()

        cv2.putText(
            display,
            "Kliknij na kolor skory. ENTER = zatwierdz, ESC = wyjdz",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        if selected_hsv["value"] is not None:
            cv2.putText(
                display,
                f"HSV: {selected_hsv['value']}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        if key == 13 and selected_hsv["value"] is not None:
            break

        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Przerwano wybieranie koloru.")

    cv2.destroyWindow(window_name)

    return selected_hsv["value"]


# ============================================================
# TRYB 2: ODEJMOWANIE TŁA
# ============================================================

def prepare_frame_for_background(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if BACKGROUND_BLUR_SIZE > 1:
        gray = cv2.GaussianBlur(
            gray,
            (BACKGROUND_BLUR_SIZE, BACKGROUND_BLUR_SIZE),
            0
        )

    return gray


def capture_background(cap):
    print("Kalibracja tła.")
    print("Usuń dłoń z kadru i nie ruszaj kamerą.")

    background_acc = None
    collected = 0

    while collected < BACKGROUND_FRAMES:
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Nie udało się pobrać klatki tła.")

        if USE_FIXED_ROI:
            frame_for_background = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        else:
            frame_for_background = frame

        gray = prepare_frame_for_background(frame_for_background)

        if background_acc is None:
            background_acc = gray.astype(np.float32)
        else:
            cv2.accumulateWeighted(gray, background_acc, 0.1)

        collected += 1

        display = frame.copy()

        if USE_FIXED_ROI:
            cv2.rectangle(
                display,
                (ROI_X1, ROI_Y1),
                (ROI_X2, ROI_Y2),
                (255, 0, 0),
                2
            )

        cv2.putText(
            display,
            f"Kalibracja tla: {collected}/{BACKGROUND_FRAMES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            "Nie pokazuj dloni podczas kalibracji",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow("Camera", display)

        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord("q"):
            raise RuntimeError("Przerwano kalibrację tła.")

    background = cv2.convertScaleAbs(background_acc)

    print("Tło zapisane.")
    print("Rozmiar tła:", background.shape)

    return background


def make_background_subtraction_mask(frame_bgr, background_gray):
    gray = prepare_frame_for_background(frame_bgr)

    diff = cv2.absdiff(background_gray, gray)

    _, mask = cv2.threshold(
        diff,
        BACKGROUND_DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    mask = clean_mask(mask)

    return mask


# ============================================================
# WSPÓLNA CZĘŚĆ: MASKA -> ROI -> FEATURE EXTRACTOR -> MODEL
# ============================================================
def fill_holes(mask):
    h, w = mask.shape[:2]

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)

    filled = mask | flood_inv

    return filled

def clean_mask(mask):
    mask = cv2.medianBlur(mask, 5)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=3)

    mask = fill_holes(mask)

    return mask

def get_largest_contour_mask(mask):
    contours = find_contours(mask)

    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < MIN_CONTOUR_AREA:
        return None, None

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [cnt], -1, 255, thickness=cv2.FILLED)

    return clean, cnt


def resize_with_padding(binary_roi, output_size=256):
    h, w = binary_roi.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Puste ROI.")

    scale = output_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        binary_roi,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


def crop_and_resize_hand_mask(mask, contour):
    x, y, w, h = cv2.boundingRect(contour)

    x1 = max(x - ROI_MARGIN, 0)
    y1 = max(y - ROI_MARGIN, 0)
    x2 = min(x + w + ROI_MARGIN, mask.shape[1])
    y2 = min(y + h + ROI_MARGIN, mask.shape[0])

    roi = mask[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError("Puste ROI dłoni.")

    roi = resize_with_padding(roi, OUTPUT_SIZE)

    return roi, (x1, y1, x2, y2)


def extract_features_from_camera_mask(mask):
    clean_mask_only_largest, contour = get_largest_contour_mask(mask)

    if clean_mask_only_largest is None:
        raise ValueError("Nie znaleziono dłoni na masce.")

    roi_mask, bbox = crop_and_resize_hand_mask(clean_mask_only_largest, contour)

    extractor = FeatureExtractor(roi_mask)
    vis_results, defect_list = extractor.process()

    features = extract_numeric_features(
        binary=vis_results["binary"],
        defect_list=defect_list
    )

    return features, roi_mask, bbox, vis_results


def draw_prediction(frame, prediction, distances, bbox=None):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Mode: {SEGMENTATION_MODE}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Gesture: {prediction}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    y = 100

    for label, distance in sorted(distances.items(), key=lambda item: item[1]):
        cv2.putText(
            frame,
            f"{label}: {distance:.3f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y += 30


def create_mask_for_current_mode(frame, selected_hsv, background_gray):
    if SEGMENTATION_MODE == "color":
        return make_skin_mask(frame, selected_hsv)

    if SEGMENTATION_MODE == "background":
        return make_background_subtraction_mask(frame, background_gray)

    raise ValueError(
        "Niepoprawny SEGMENTATION_MODE. Użyj 'color' albo 'background'."
    )


def main():
    database = load_database()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Nie udało się otworzyć kamery.")

    selected_hsv = None
    background_gray = None

    if SEGMENTATION_MODE == "color":
        selected_hsv = choose_color_from_camera(cap)

    elif SEGMENTATION_MODE == "background":
        background_gray = capture_background(cap)

    else:
        cap.release()
        raise ValueError(
            "Niepoprawny SEGMENTATION_MODE. Ustaw 'color' albo 'background'."
        )

    print("Start rozpoznawania.")
    print("ESC albo Q = wyjście.")
    print("R = ponowna kalibracja aktualnego trybu.")
    print("W trybie color: R wybiera kolor od nowa.")
    print("W trybie background: R zapisuje tło od nowa.")

    last_prediction = "UNKNOWN"
    last_distances = {}
    
    # debug
    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Nie udało się pobrać klatki.")
            break

        frame_counter = 0

        if USE_FIXED_ROI:
            work_frame = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            mask = create_mask_for_current_mode(work_frame, selected_hsv, background_gray)

            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
        else:
            work_frame = frame
            mask = create_mask_for_current_mode(frame, selected_hsv, background_gray)


        try:
            features, roi_mask, bbox, vis_results = extract_features_from_camera_mask(mask)

            print_features(features, database, frame_counter)
            
            prediction, distances = predict_by_centroid(features, database)

            last_prediction = prediction
            last_distances = distances

            if USE_FIXED_ROI and bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_to_draw = (
                    x1 + ROI_X1,
                    y1 + ROI_Y1,
                    x2 + ROI_X1,
                    y2 + ROI_Y1
                )
            else:
                bbox_to_draw = bbox

            draw_prediction(frame, prediction, distances, bbox_to_draw)

            cv2.imshow("ROI mask 256x256", roi_mask)
            cv2.imshow("Feature binary", vis_results["binary"])
            cv2.imshow("Feature defects", vis_results["defect_vis"])

        except Exception:
            draw_prediction(frame, last_prediction, last_distances)

        cv2.imshow("Camera", frame)
        cv2.imshow("Binary mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

        if key == ord("r"):
            cv2.destroyAllWindows()

            if SEGMENTATION_MODE == "color":
                selected_hsv = choose_color_from_camera(cap)

            elif SEGMENTATION_MODE == "background":
                background_gray = capture_background(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()