from pathlib import Path

import cv2

from feature_extractor import FeatureExtractor


INPUT_DIR = Path("masks_cropped")
OUTPUT_DIR = Path("masks_cropped_hull")

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def find_mask_files(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder wejściowy nie istnieje: {input_dir}")

    files = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"Nie znaleziono masek w folderze: {input_dir}")

    return sorted(files)


def get_defect_visualization(vis: dict):
    """
    Obsługuje oba warianty nazw, bo w Twoim kodzie pojawiały się wersje:
    - vis["defects"]
    - vis["defect_vis"]
    """

    preferred_keys = ["defects", "defect_vis"]

    for key in preferred_keys:
        if key in vis and vis[key] is not None:
            return vis[key]

    available = ", ".join(vis.keys())
    raise KeyError(
        "FeatureExtractor.process() nie zwrócił wizualizacji defektów. "
        f"Dostępne klucze: {available}"
    )


def process_one_mask(mask_path: Path, input_dir: Path, output_dir: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Nie udało się wczytać maski: {mask_path}")

    extractor = FeatureExtractor(
        image=mask,
        mask_name=mask_path.name
    )

    vis, features = extractor.process()

    output_image = get_defect_visualization(vis)

    relative_path = mask_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), output_image)

    if not success:
        raise ValueError(f"Nie udało się zapisać pliku: {output_path}")

    return output_path, features


def export_hull_and_defects(
    input_dir: str | Path = INPUT_DIR,
    output_dir: str | Path = OUTPUT_DIR
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = find_mask_files(input_dir)

    saved_count = 0
    failed = []

    for mask_path in mask_files:
        try:
            output_path, features = process_one_mask(
                mask_path=mask_path,
                input_dir=input_dir,
                output_dir=output_dir
            )

            saved_count += 1
            print(f"[OK] {mask_path} -> {output_path}")

        except Exception as error:
            failed.append((mask_path, error))
            print(f"[ERROR] {mask_path}: {error}")

    print("=" * 60)
    print(f"Zapisano plików: {saved_count}")
    print(f"Błędów: {len(failed)}")
    print(f"Folder wynikowy: {output_dir}")

    if failed:
        print("=" * 60)
        print("Pliki z błędami:")
        for path, error in failed:
            print(f"- {path}: {error}")

    return saved_count, failed


if __name__ == "__main__":
    export_hull_and_defects()
