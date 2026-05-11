from pathlib import Path
import json
import cv2
import numpy as np

from feature_extractor import FeatureExtractor


MASKS_DIR = Path("masks_cropped")
OUTPUT_DIR = Path("features_database")

DATABASE_PATH = OUTPUT_DIR / "features_database.txt"
PER_FILE_SUMMARY_PATH = OUTPUT_DIR / "per_file_features.txt"
SAMPLES_DIR = OUTPUT_DIR / "samples"

OUTPUT_SIZE = 256


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


def get_label_from_filename(filename):
    name = filename.upper()

    if name.startswith("1"):
        return "LIKE"

    if name.startswith("L"):
        return "L"

    return None


def find_contours(binary_img):
    found = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return contours


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


def features_to_vector(features):
    return [float(features[name]) for name in FEATURE_NAMES]


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Nie udało się wczytać maski: {path}")

    mask = cv2.resize(mask, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)

    return mask


def calculate_global_scaler(vectors):
    vectors = np.array(vectors, dtype=np.float32)

    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)

    std[std == 0] = 1.0

    return mean, std


def scale_vector(vector, mean, std):
    vector = np.array(vector, dtype=np.float32)
    return ((vector - mean) / std).tolist()


def build_templates(samples, global_mean, global_std):
    labels = sorted(set(sample["label"] for sample in samples))
    templates = {}

    for label in labels:
        label_vectors = [
            sample["vector"]
            for sample in samples
            if sample["label"] == label
        ]

        label_vectors_np = np.array(label_vectors, dtype=np.float32)
        mean_vector = label_vectors_np.mean(axis=0)

        templates[label] = {
            "count": len(label_vectors),
            "mean_vector": mean_vector.tolist(),
            "mean_vector_scaled": scale_vector(mean_vector, global_mean, global_std),
        }

    return templates


def save_single_sample(sample):
    output_path = SAMPLES_DIR / f"{Path(sample['file']).stem}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=4, ensure_ascii=False)


def build_database():
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(exist_ok=True)

    files = []

    for ext in ["*.bmp", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(MASKS_DIR.glob(ext))

    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError(f"Brak plików w folderze: {MASKS_DIR}")

    samples = []
    errors = []

    for path in files:
        label = get_label_from_filename(path.name)

        if label is None:
            errors.append({
                "file": path.name,
                "error": "Nieznana etykieta. Plik musi zaczynać się od '1' albo 'L'."
            })
            continue

        try:
            mask = load_mask(path)

            extractor = FeatureExtractor(mask)
            vis_results, defect_list = extractor.process()

            features = extract_numeric_features(
                binary=vis_results["binary"],
                defect_list=defect_list
            )

            vector = features_to_vector(features)

            sample = {
                "file": path.name,
                "label": label,
                "features": features,
                "vector": vector,
            }

            samples.append(sample)
            save_single_sample(sample)

            print(f"OK: {path.name} -> {label}")

        except Exception as e:
            errors.append({
                "file": path.name,
                "error": str(e)
            })
            print(f"BŁĄD: {path.name} -> {e}")

    if len(samples) == 0:
        raise ValueError("Nie udało się zapisać żadnych cech.")

    vectors = [sample["vector"] for sample in samples]

    global_mean, global_std = calculate_global_scaler(vectors)

    for sample in samples:
        sample["vector_scaled"] = scale_vector(
            sample["vector"],
            global_mean,
            global_std
        )

    templates = build_templates(samples, global_mean, global_std)

    database = {
        "feature_names": FEATURE_NAMES,
        "output_size": OUTPUT_SIZE,
        "labels": {
            "1": "LIKE",
            "L": "L"
        },
        "scaler": {
            "mean": global_mean.tolist(),
            "std": global_std.tolist()
        },
        "samples": samples,
        "templates": templates,
        "errors": errors
    }

    with open(DATABASE_PATH, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

    with open(PER_FILE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(f"file={sample['file']}; label={sample['label']}; ")

            feature_text = []
            for name in FEATURE_NAMES:
                feature_text.append(f"{name}={sample['features'][name]}")

            f.write("; ".join(feature_text))
            f.write("\n")

    print()
    print("Zapisano bazę cech:")
    print(DATABASE_PATH)
    print()
    print("Zapisano cechy każdego pliku:")
    print(PER_FILE_SUMMARY_PATH)
    print()
    print("Zapisano osobne pliki próbek:")
    print(SAMPLES_DIR)
    print()
    print("Liczba poprawnych próbek:", len(samples))
    print("Liczba błędów:", len(errors))

    for label, template in templates.items():
        print(f"{label}: {template['count']} próbek")


def load_database(path=DATABASE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def euclidean_distance(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def predict_by_centroid(features, database):
    vector = features_to_vector(features)

    mean = np.array(database["scaler"]["mean"], dtype=np.float32)
    std = np.array(database["scaler"]["std"], dtype=np.float32)

    vector_scaled = scale_vector(vector, mean, std)

    distances = {}

    for label, template in database["templates"].items():
        distance = euclidean_distance(
            vector_scaled,
            template["mean_vector_scaled"]
        )
        distances[label] = distance

    predicted_label = min(distances, key=distances.get)

    return predicted_label, distances


def predict_by_nearest_sample(features, database):
    vector = features_to_vector(features)

    mean = np.array(database["scaler"]["mean"], dtype=np.float32)
    std = np.array(database["scaler"]["std"], dtype=np.float32)

    vector_scaled = scale_vector(vector, mean, std)

    best_label = None
    best_file = None
    best_distance = None

    distances = []

    for sample in database["samples"]:
        distance = euclidean_distance(
            vector_scaled,
            sample["vector_scaled"]
        )

        distances.append({
            "file": sample["file"],
            "label": sample["label"],
            "distance": distance
        })

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = sample["label"]
            best_file = sample["file"]

    return best_label, best_file, best_distance, distances


def extract_features_from_mask(mask):
    mask = cv2.resize(mask, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)

    extractor = FeatureExtractor(mask)
    vis_results, defect_list = extractor.process()

    features = extract_numeric_features(
        binary=vis_results["binary"],
        defect_list=defect_list
    )

    return features


if __name__ == "__main__":
    build_database()