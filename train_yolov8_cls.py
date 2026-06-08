from ultralytics import YOLO
import time
from pathlib import Path


DATASET_DIR = "dataset_yolo_split"
PROJECT_DIR = "runs/classify"
MODEL_NAME = "hgr_yolov8n_cls"
EPOCHS = 80


def main():
    train_start = time.perf_counter()

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=DATASET_DIR,
        epochs=EPOCHS,
        imgsz=224,
        batch=32,
        patience=15,
        project=PROJECT_DIR,
        name=MODEL_NAME,
        pretrained=True,
        device="cpu"
    )

    train_end = time.perf_counter()
    training_time = train_end - train_start

    best_model_path = Path(PROJECT_DIR) / MODEL_NAME / "weights" / "best.pt"

    print("\nTrening zakończony.")
    print("Najlepszy model:")
    print(best_model_path)

    print("\nCzasy:")
    print(f"Suma czasu treningu: {training_time:.4f} s")
    print(f"Średni czas treningu jednej epoki: {training_time / EPOCHS:.4f} s")

    with open("yolov8_training_time.txt", "w", encoding="utf-8") as file:
        file.write(str(training_time))


if __name__ == "__main__":
    main()