from ultralytics import YOLO


MODEL_PATH = "runs/classify/runs/classify/hgr_yolov8n_cls/weights/best.pt"
DATASET_DIR = "dataset_yolo_split"


def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATASET_DIR,
        split="test",
        imgsz=224
    )

    print("Top-1 accuracy:", metrics.top1)
    print("Top-5 accuracy:", metrics.top5)


if __name__ == "__main__":
    main()