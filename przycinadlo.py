from pathlib import Path
import cv2
import sys


# Wszystkie oryginalne zdjęcia JPG
ALL_ORIGINALS_FOLDER = Path(
    r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\experiment_1"
)

# Wszystkie maski BMP
ALL_MASKS_FOLDER = Path(
    r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\experiment_2"
)

ALREADY_CROPPED_ORIGINALS_FOLDER = Path(
    r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\original_cropped"
)

OUTPUT_ORIGINALS_FOLDER = Path(
    r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\original_cropped_new"
)

OUTPUT_MASKS_FOLDER = Path(
    r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\masks_cropped_new"
)

OUTPUT_ORIGINALS_FOLDER = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\nowe_obrazy_oryginalne")
OUTPUT_MASKS_FOLDER = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\nowe_obrazy__maski")

OUTPUT_ORIGINALS_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_FOLDER.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".bmp", ".png"]


def get_file_stems(folder: Path):
    if not folder.exists():
        return set()

    return {
        path.stem
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def get_original_files(folder: Path):
    return sorted([
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg"]
    ])


# ============================================================
# SKANOWANIE PLIKÓW
# ============================================================

all_original_files = get_original_files(ALL_ORIGINALS_FOLDER)

if not all_original_files:
    print("Brak plików JPG/JPEG w folderze ze wszystkimi oryginałami.")
    sys.exit()


already_cropped_stems = get_file_stems(ALREADY_CROPPED_ORIGINALS_FOLDER)

# Dodatkowo skrypt pomija też te, które już są w nowym output folderze,
# żeby po ponownym uruchomieniu nie przycinać drugi raz tego samego.
already_output_stems = get_file_stems(OUTPUT_ORIGINALS_FOLDER)

processed_stems = already_cropped_stems | already_output_stems

files_to_crop = [
    path for path in all_original_files
    if path.stem not in processed_stems
]


print(f"Wszystkie oryginały: {len(all_original_files)}")
print(f"Już przycięte:       {len(already_cropped_stems)}")
print(f"Już w output:        {len(already_output_stems)}")
print(f"Do przycięcia:       {len(files_to_crop)}")

if not files_to_crop:
    print("Nie ma nic do przycinania.")
    sys.exit()


print("\nSterowanie:")
print(" - zaznacz prostokąt myszką")
print(" - ENTER / SPACJA = zatwierdź")
print(" - C = anuluj zaznaczenie")
print(" - ESC = pomiń aktualny obraz")


# ============================================================
# PRZETWARZANIE
# ============================================================

for original_path in files_to_crop:
    stem = original_path.stem
    mask_path = ALL_MASKS_FOLDER / f"{stem}.bmp"

    print("\n-----------------------------------")
    print(f"Oryginał: {original_path.name}")
    print(f"Maska:    {mask_path.name}")

    if not mask_path.exists():
        print("Brak odpowiadającej maski BMP — pomijam.")
        continue

    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if original is None:
        print("Nie udało się wczytać oryginału — pomijam.")
        continue

    if mask is None:
        print("Nie udało się wczytać maski — pomijam.")
        continue

    if original.shape[:2] != mask.shape[:2]:
        print("Oryginał i maska mają różne rozmiary — pomijam.")
        print(f"Rozmiar oryginału: {original.shape[:2]}")
        print(f"Rozmiar maski:     {mask.shape[:2]}")
        continue

    # ========================================================
    # JEDNO ZAZNACZENIE DLA ORYGINAŁU I MASKI
    # ========================================================

    window_name = f"Zaznacz ROI: {original_path.name}"

    roi = cv2.selectROI(
        window_name,
        original,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("Nie zaznaczono obszaru — pomijam.")
        continue

    cropped_original = original[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]

    output_original_path = OUTPUT_ORIGINALS_FOLDER / f"{stem}.jpg"
    output_mask_path = OUTPUT_MASKS_FOLDER / f"{stem}.bmp"

    ok_original = cv2.imwrite(
        str(output_original_path),
        cropped_original,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    ok_mask = cv2.imwrite(
        str(output_mask_path),
        cropped_mask
    )

    if ok_original and ok_mask:
        print("Zapisano:")
        print(f" - oryginał: {output_original_path}")
        print(f" - maska:    {output_mask_path}")
    else:
        print("Błąd zapisu.")


cv2.destroyAllWindows()
print("\nGotowe.")