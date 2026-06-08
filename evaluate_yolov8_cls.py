from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import time


MODEL_PATH = "runs/classify/hgr_yolov8n_cls/weights/best.pt"
TEST_DIR = Path("dataset_yolo_split/test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_test_images():
    image_paths = []

    for class_dir in TEST_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        raise ValueError("Brak obrazów testowych w dataset_yolo_split/test.")

    return image_paths


def plot_confusion_matrix(true_labels, predicted_labels):
    labels = sorted(set(true_labels) | set(predicted_labels))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        row = label_to_idx[true_label]
        col = label_to_idx[predicted_label]
        matrix[row, col] += 1

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)

    plt.title(f"Macierz pomyłek YOLOv8 - test, n={matrix.sum()}")
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Prawdziwa klasa")

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main():
    model = YOLO(MODEL_PATH)

    image_paths = load_test_images()

    true_labels = []
    predicted_labels = []
    prediction_times = []

    # Rozgrzewka modelu - pierwsza predykcja często jest sztucznie wolniejsza
    model.predict(
        source=str(image_paths[0]),
        imgsz=224,
        verbose=False,
        device="cpu"
    )

    prediction_start = time.perf_counter()

    for image_path in image_paths:
        true_label = image_path.parent.name

        single_prediction_start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=224,
            verbose=False,
            device="cpu"
        )

        single_prediction_end = time.perf_counter()

        result = results[0]
        predicted_class_id = int(result.probs.top1)
        predicted_label = str(model.names[predicted_class_id])

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        prediction_times.append(single_prediction_end - single_prediction_start)

    prediction_end = time.perf_counter()

    correct_count = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )

    incorrect_count = len(true_labels) - correct_count
    accuracy = correct_count / len(true_labels) * 100

    total_prediction_time = prediction_end - prediction_start
    average_prediction_time = total_prediction_time / len(image_paths)

    print("\nWyniki YOLOv8:")
    print(f"Liczba obrazów testowych: {len(image_paths)}")
    print(f"Poprawne klasyfikacje: {correct_count}")
    print(f"Błędne klasyfikacje: {incorrect_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nCzasy:")
    print(f"Suma czasu predykcji: {total_prediction_time:.4f} s")
    print(f"Średni czas predykcji jednego obrazu: {average_prediction_time:.6f} s")
    print(f"Najkrótsza predykcja jednego obrazu: {min(prediction_times):.6f} s")
    print(f"Najdłuższa predykcja jednego obrazu: {max(prediction_times):.6f} s")

    if Path("yolov8_training_time.txt").exists():
        with open("yolov8_training_time.txt", "r", encoding="utf-8") as file:
            training_time = float(file.read())

        print(f"Suma czasu treningu: {training_time:.4f} s")

    plot_confusion_matrix(true_labels, predicted_labels)


if __name__ == "__main__":
    main()