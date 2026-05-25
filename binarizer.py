import cv2
import numpy as np


def get_mask_from_range(lower_hsv, upper_hsv, image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array(lower_hsv, dtype=np.uint8)
    upper_hsv = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.bitwise_not(mask)

    return mask


def get_range_from_mask(image_bgr, mask):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hand_area = mask == 0

    hand_area_uint8 = hand_area.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    hand_area_uint8 = cv2.erode(hand_area_uint8, kernel, iterations=1)
    hand_area = hand_area_uint8 > 0

    selected_pixels = image_hsv[hand_area]
    selected_pixels = selected_pixels[selected_pixels[:, 2] > 40]

    if selected_pixels.size == 0:
        raise ValueError("Maska nie zawiera pikseli dłoni.")

    lower_hsv = np.percentile(selected_pixels, 5, axis=0)
    upper_hsv = np.percentile(selected_pixels, 95, axis=0)

    lower_hsv = np.array(lower_hsv, dtype=np.uint8)
    upper_hsv = np.array(upper_hsv, dtype=np.uint8)

    return lower_hsv, upper_hsv
