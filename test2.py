import cv2
import numpy as np


def normalize_hand(image, hand_mask, contour, output_size=256):
    x, y, w, h = cv2.boundingRect(contour)

    margin = 10
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, image.shape[1])
    y2 = min(y + h + margin, image.shape[0])

    hand_crop = image[y1:y2, x1:x2]
    mask_crop = hand_mask[y1:y2, x1:x2]

    hand_only = cv2.bitwise_and(hand_crop, hand_crop, mask=mask_crop)

    crop_h, crop_w = mask_crop.shape
    scale = min(output_size / crop_w, output_size / crop_h)

    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))

    resized_img = cv2.resize(hand_only, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    canvas_mask = np.zeros((output_size, output_size), dtype=np.uint8)

    ox = (output_size - new_w) // 2
    oy = (output_size - new_h) // 2

    canvas_img[oy:oy + new_h, ox:ox + new_w] = resized_img
    canvas_mask[oy:oy + new_h, ox:ox + new_w] = resized_mask

    return canvas_img, canvas_mask


def grabcut_hand_segmentation(image, rect, iterations=5):
    mask = np.zeros(image.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_RECT
    )

    # 0 i 2 = tło, 1 i 3 = obiekt
    binary_mask = np.where((mask == 0) | (mask == 2), 0, 255).astype("uint8")

    return binary_mask


def clean_mask(mask):
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def get_largest_contour(mask, min_area=1500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < min_area:
        return None

    return largest


def main():
    image_path = r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\cropped\L_P_hgr1_id08_3.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print("Nie udało się wczytać obrazu.")
        return

    # Prostokąt obejmujący dłoń: (x, y, w, h)
    # Musi zawierać całą dłoń i jak najmniej tła
    rect = (0+1 , 0+1, image.shape[1]-1, image.shape[0]-1)

    raw_mask = grabcut_hand_segmentation(image, rect, iterations=5)
    cleaned_mask = clean_mask(raw_mask)

    contour = get_largest_contour(cleaned_mask)

    hand_mask = np.zeros_like(cleaned_mask)
    contour_view = image.copy()
    rect_view = image.copy()

    x, y, w, h = rect
    cv2.rectangle(rect_view, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if contour is None:
        print("Nie wykryto dłoni.")
        cv2.imshow("1. Original", image)
        cv2.imshow("2. GrabCut rect", rect_view)
        cv2.imshow("3. Raw mask", raw_mask)
        cv2.imshow("4. Cleaned mask", cleaned_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cv2.drawContours(hand_mask, [contour], -1, 255, -1)
    cv2.drawContours(contour_view, [contour], -1, (0, 255, 0), 2)

    bx, by, bw, bh = cv2.boundingRect(contour)
    cv2.rectangle(contour_view, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    segmented_hand = cv2.bitwise_and(image, image, mask=hand_mask)
    normalized_hand, normalized_mask = normalize_hand(image, hand_mask, contour, output_size=256)

    cv2.imshow("1. Original", image)
    cv2.imshow("2. GrabCut rect", rect_view)
    cv2.imshow("3. Raw mask", raw_mask)
    cv2.imshow("4. Cleaned mask", cleaned_mask)
    cv2.imshow("5. Final hand mask", hand_mask)
    cv2.imshow("6. Contour + bounding box", contour_view)
    cv2.imshow("7. Segmented hand", segmented_hand)
    cv2.imshow("8. Normalized hand", normalized_hand)
    cv2.imshow("9. Normalized mask", normalized_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()