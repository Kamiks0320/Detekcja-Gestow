import cv2

import numpy as np
def fill_holes(mask):
    filled = mask.copy()
    h, w = mask.shape[:2]

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)

    filled_inv = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(mask, filled_inv)

    return result

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


image_path = r"C:\Users\marci\Documents\studium\semestr6\Wizja Komputerowa\Projekt\Coding\cropped\L_P_hgr1_id11_1.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Nie udało się wczytać obrazu")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 10, 60)    # bardzo czułe
#edges = cv2.Canny(gray, 100, 200)
# binaryzacja krawędzi
_, edge_mask = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

# morfologia żeby zamknąć obiekt
kernel = np.ones((5, 5), np.uint8)
edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hand_mask = np.zeros_like(edge_mask)

vis = image.copy()

best_contour = None
best_area = 0

for c in contours:
    area = cv2.contourArea(c)
    if area < 3000:
        continue

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h != 0 else 0

    if 0.4 < aspect_ratio < 1.8:
        if area > best_area:
            best_area = area
            best_contour = c

if best_contour is not None:
    cv2.drawContours(hand_mask, [best_contour], -1, 255, -1)
    hand_mask = fill_holes(hand_mask)
    cv2.drawContours(vis, [best_contour], -1, (0, 255, 0), 2)

    bx, by, bw, bh = cv2.boundingRect(best_contour)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    normalized_hand, normalized_mask = normalize_hand(image, hand_mask, best_contour)

    cv2.imshow("Normalized hand", normalized_hand)
    cv2.imshow("Normalized mask", normalized_mask)

cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edges", edges)
cv2.imshow("Edge mask", edge_mask)
cv2.imshow("Hand mask", hand_mask)
cv2.imshow("Contour", vis)

cv2.waitKey(0)
cv2.destroyAllWindows()