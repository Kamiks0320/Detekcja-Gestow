from pathlib import Path
import shutil
import random

SOURCE_DIR = Path("dataset_yolo")
OUTPUT_DIR = Path("dataset_yolo_split")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)


def clear_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def copy_images(images, split_name, class_name):
    target_dir = OUTPUT_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, target_dir / image_path.name)


def main():
    clear_output()

    class_dirs = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError("Brak folderów klas w dataset_yolo.")

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

        copy_images(train_images, "train", class_name)
        copy_images(val_images, "val", class_name)
        copy_images(test_images, "test", class_name)

        print(
            f"Klasa {class_name}: "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print(f"\nGotowy dataset zapisany w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()