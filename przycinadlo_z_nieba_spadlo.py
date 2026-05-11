from pathlib import Path
import cv2


# ============================================================
# USTAW ŚCIEŻKI
# ============================================================

ORIGINALS_FOLDER = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\original_images")
MASKS_FOLDER = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\masks")

OUTPUT_ORIGINALS_FOLDER = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\original_cropped")
OUTPUT_MASKS_FOLDER  = Path(r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\Detekcja-Gestow\masks_cropped")

OUTPUT_ORIGINALS_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS_FOLDER .mkdir(parents=True, exist_ok=True)

# ============================================================
# POBRANIE LISTY ORYGINAŁÓW JPG/JPEG
# ============================================================

original_files = sorted([
    path for path in ORIGINALS_FOLDER.iterdir()
    if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg"]
])

if not original_files:
    print("Brak plików JPG/JPEG w folderze z oryginałami.")
    sys.exit()


print(f"Znaleziono {len(original_files)} oryginałów.")
print("\nSterowanie w oknie selectROI:")
print(" - zaznacz prostokąt myszką")
print(" - ENTER albo SPACJA = zatwierdzenie")
print(" - C = anulowanie zaznaczenia")
print(" - jeśli nic nie zaznaczysz i zatwierdzisz, plik zostanie pominięty")


# ============================================================
# PRZETWARZANIE
# ============================================================

for original_path in original_files:
    stem = original_path.stem
    mask_path = MASKS_FOLDER / f"{stem}.bmp"

    print("\n-----------------------------------")
    print(f"Plik oryginału: {original_path.name}")
    print(f"Szukana maska:  {mask_path.name}")

    if not mask_path.exists():
        print("Brak odpowiadającej maski .bmp — pomijam.")
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
    # ZAZNACZASZ TYLKO RAZ NA ORYGINALE
    # ========================================================
    window_name = f"Zaznacz ROI: {original_path.name}"
    roi = cv2.selectROI(window_name, original, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("Nie zaznaczono obszaru — pomijam.")
        continue

    # ========================================================
    # TEN SAM PROSTOKĄT DLA ORYGINAŁU I MASKI
    # ========================================================
    cropped_original = original[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    output_original_path = OUTPUT_ORIGINALS_FOLDER / f"{stem}.jpg"
    output_mask_path = OUTPUT_MASKS_FOLDER / f"{stem}.bmp"

    ok1 = cv2.imwrite(
        str(output_original_path),
        cropped_original,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    ok2 = cv2.imwrite(str(output_mask_path), cropped_mask)

    if ok1 and ok2:
        print("Zapisano:")
        print(f" - oryginał: {output_original_path}")
        print(f" - maska:    {output_mask_path}")
    else:
        print("Błąd przy zapisie plików.")

cv2.destroyAllWindows()
print("\nGotowe.")