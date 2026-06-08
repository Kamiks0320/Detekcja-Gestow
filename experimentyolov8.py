from pathlib import Path
import shutil
import random
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# =========================
# KONFIGURACJA
# =========================

SOURCE_DIR = Path("dataset_yolo")

SPLITS_OUTPUT_DIR = Path("dataset_yolo_experiments")
RUNS_PROJECT_DIR = "classify_yolo_experiments"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

REPEATS = 5
BASE_SEED = 42

EPOCHS = 80
IMGSZ = 224
BATCH = 32
PATIENCE = 15
DEVICE = "cpu"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# SPLIT DANYCH
# =========================

def clear_dir(path):
    if path.exists():
        shutil.rmtree(path)


def copy_images(images, split_name, class_name, output_dir):
    target_dir = output_dir / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, target_dir / image_path.name)


def create_split(output_dir, seed):
    clear_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    class_dirs = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError("Brak folderów klas w dataset_yolo.")

    split_info = {}

    for class_dir in class_dirs:
        class_name = class_dir.name

        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < 3:
            raise ValueError(f"Za mało zdjęć w klasie {class_name}. Minimum sensowne to 3.")

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        copy_images(train_images, "train", class_name, output_dir)
        copy_images(val_images, "val", class_name, output_dir)
        copy_images(test_images, "test", class_name, output_dir)

        split_info[class_name] = {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }

    return split_info


# =========================
# TEST MODELU
# =========================

def load_test_images(test_dir):
    image_paths = []

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        raise ValueError(f"Brak obrazów testowych w: {test_dir}")

    return image_paths


def evaluate_model(model_path, test_dir):
    model = YOLO(model_path)
    image_paths = load_test_images(test_dir)

    true_labels = []
    predicted_labels = []
    prediction_times = []

    # Rozgrzewka modelu, żeby pierwsza predykcja nie zaburzała wyniku
    model.predict(
        source=str(image_paths[0]),
        imgsz=IMGSZ,
        verbose=False,
        device=DEVICE
    )

    test_start = time.perf_counter()

    for image_path in image_paths:
        true_label = image_path.parent.name

        single_start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            verbose=False,
            device=DEVICE
        )

        single_end = time.perf_counter()

        result = results[0]
        predicted_class_id = int(result.probs.top1)
        predicted_label = str(model.names[predicted_class_id])

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        prediction_times.append(single_end - single_start)

    test_end = time.perf_counter()

    correct_count = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )

    incorrect_count = len(true_labels) - correct_count
    accuracy = correct_count / len(true_labels) * 100

    total_test_time = test_end - test_start
    avg_prediction_time = total_test_time / len(image_paths)

    return {
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "test_images": len(image_paths),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy": accuracy,
        "test_time": total_test_time,
        "avg_prediction_time": avg_prediction_time,
        "min_prediction_time": min(prediction_times),
        "max_prediction_time": max(prediction_times)
    }


# =========================
# MACIERZ POMYŁEK
# =========================

def plot_confusion_matrix(true_labels, predicted_labels, title, output_path):
    labels = sorted(set(true_labels) | set(predicted_labels))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        row = label_to_idx[true_label]
        col = label_to_idx[predicted_label]
        matrix[row, col] += 1

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)

    plt.title(title)
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Prawdziwa klasa")

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# GŁÓWNY EKSPERYMENT
# =========================

def main():
    all_results = []

    all_true_labels = []
    all_predicted_labels = []

    print("Start eksperymentu YOLOv8")
    print(f"Liczba powtórzeń: {REPEATS}")

    for repeat in range(1, REPEATS + 1):
        seed = BASE_SEED + repeat
        split_dir = SPLITS_OUTPUT_DIR / f"split_{repeat}"

        print("\n" + "=" * 60)
        print(f"EKSPERYMENT {repeat}/{REPEATS}")
        print(f"Seed splitu: {seed}")

        split_info = create_split(split_dir, seed)

        print("\nPodział danych:")
        for class_name, counts in split_info.items():
            print(
                f"{class_name}: "
                f"train={counts['train']}, "
                f"val={counts['val']}, "
                f"test={counts['test']}"
            )

        run_name = f"hgr_yolov8n_cls_exp_{repeat}"

        model = YOLO("yolov8n-cls.pt")

        train_start = time.perf_counter()

        model.train(
            data=str(split_dir),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            project=RUNS_PROJECT_DIR,
            name=run_name,
            pretrained=True,
            device=DEVICE,
            exist_ok=True
        )

        train_end = time.perf_counter()
        train_time = train_end - train_start

        # YOLO samo mówi, gdzie faktycznie zapisało wyniki
        save_dir = Path(model.trainer.save_dir)
        best_model_path = save_dir / "weights" / "best.pt"

        if not best_model_path.exists():
            best_model_path = save_dir / "weights" / "last.pt"

        if not best_model_path.exists():
            raise FileNotFoundError(f"Nie znaleziono modelu ani best.pt, ani last.pt w: {save_dir}")

        print(f"Używany model do testu: {best_model_path}")

        eval_result = evaluate_model(
            model_path=best_model_path,
            test_dir=split_dir / "test"
        )

        all_true_labels.extend(eval_result["true_labels"])
        all_predicted_labels.extend(eval_result["predicted_labels"])

        result_row = {
            "repeat": repeat,
            "seed": seed,
            "train_time": train_time,
            "test_time": eval_result["test_time"],
            "avg_prediction_time": eval_result["avg_prediction_time"],
            "test_images": eval_result["test_images"],
            "correct": eval_result["correct"],
            "incorrect": eval_result["incorrect"],
            "accuracy": eval_result["accuracy"]
        }

        all_results.append(result_row)

        print("\nWyniki eksperymentu:")
        print(f"Liczba obrazów testowych: {eval_result['test_images']}")
        print(f"Poprawne klasyfikacje: {eval_result['correct']}")
        print(f"Błędne klasyfikacje: {eval_result['incorrect']}")
        print(f"Accuracy: {eval_result['accuracy']:.2f}%")

        print("\nCzasy:")
        print(f"Czas treningu: {train_time:.4f} s")
        print(f"Czas testu/predykcji: {eval_result['test_time']:.4f} s")
        print(f"Średni czas predykcji jednego obrazu: {eval_result['avg_prediction_time']:.6f} s")

    # =========================
    # PODSUMOWANIE
    # =========================

    train_times = [r["train_time"] for r in all_results]
    test_times = [r["test_time"] for r in all_results]
    avg_prediction_times = [r["avg_prediction_time"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]

    total_correct = sum(r["correct"] for r in all_results)
    total_incorrect = sum(r["incorrect"] for r in all_results)
    total_test_images = sum(r["test_images"] for r in all_results)

    global_accuracy = total_correct / total_test_images * 100

    print("\n" + "=" * 60)
    print("PODSUMOWANIE 5 EKSPERYMENTÓW YOLOv8")
    print("=" * 60)

    print(f"Liczba wszystkich klasyfikacji testowych: {total_test_images}")
    print(f"Suma poprawnych klasyfikacji: {total_correct}")
    print(f"Suma błędnych klasyfikacji: {total_incorrect}")

    print(f"\nŚrednia liczba poprawnych klasyfikacji: {total_correct / REPEATS:.2f}")
    print(f"Średnia liczba błędnych klasyfikacji: {total_incorrect / REPEATS:.2f}")

    print(f"\nŚrednie accuracy z eksperymentów: {np.mean(accuracies):.2f}%")
    print(f"Globalne accuracy ze wszystkich predykcji: {global_accuracy:.2f}%")

    print("\nCzasy:")
    print(f"Średni czas treningu: {np.mean(train_times):.4f} s")
    print(f"Średni czas testu/predykcji: {np.mean(test_times):.4f} s")
    print(f"Średni czas predykcji jednego obrazu: {np.mean(avg_prediction_times):.6f} s")

    print(f"\nNajkrótszy czas treningu: {min(train_times):.4f} s")
    print(f"Najdłuższy czas treningu: {max(train_times):.4f} s")

    # =========================
    # ZAPIS CSV
    # =========================

    csv_path = "yolov8_experiment_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "repeat",
                "seed",
                "train_time",
                "test_time",
                "avg_prediction_time",
                "test_images",
                "correct",
                "incorrect",
                "accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nWyniki zapisano do pliku: {csv_path}")

    # =========================
    # MACIERZ POMYŁEK ZE WSZYSTKICH 5 TESTÓW
    # =========================

    plot_confusion_matrix(
        true_labels=all_true_labels,
        predicted_labels=all_predicted_labels,
        title=f"Macierz pomyłek YOLOv8 - 5 eksperymentów, n={len(all_true_labels)}",
        output_path="yolov8_confusion_matrix_5_experiments.png"
    )

    print("Macierz pomyłek zapisana do: yolov8_confusion_matrix_5_experiments.png")


if __name__ == "__main__":
    main()