import cv2
import numpy as np

class Binarizer:
    def __init__(self, image_bgr=None):
        self.image_bgr = image_bgr

    def set_image(self, image_bgr):
        if image_bgr is None:
            raise ValueError("Obraz nie może być None.")

        self.image_bgr = image_bgr

    def get_mask_from_range(self, lower_hsv, upper_hsv, image_bgr=None, hand_as_black=True):
        if image_bgr is None:
            image_bgr = self.image_bgr

        if image_bgr is None:
            raise ValueError("Nie podano obrazu.")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        # OpenCV: piksele w zakresie HSV będą białe
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Twoje oryginalne maski mają dłoń czarną,
        # więc odwracamy wynik.
        if hand_as_black:
            mask = cv2.bitwise_not(mask)

        return mask

    def get_range_from_mask(self, image_bgr, mask):
        if image_bgr is None:
            raise ValueError("Obraz nie może być None.")

        if mask is None:
            raise ValueError("Maska nie może być None.")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Dłoń jest czarna na masce
        hand_area = mask == 0

        # Opcjonalnie: lekkie zmniejszenie obszaru, żeby nie brać krawędzi
        hand_area_uint8 = hand_area.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        hand_area_uint8 = cv2.erode(hand_area_uint8, kernel, iterations=1)
        hand_area = hand_area_uint8 > 0

        selected_pixels = hsv[hand_area]

        if selected_pixels.size == 0:
            raise ValueError("Maska nie zawiera pikseli dłoni.")

        # Odrzuć bardzo ciemne piksele, bo psują zakres
        selected_pixels = selected_pixels[selected_pixels[:, 2] > 40]

        if selected_pixels.size == 0:
            raise ValueError("Po odrzuceniu ciemnych pikseli nie zostały piksele dłoni.")

        lower_hsv = np.percentile(selected_pixels, 5, axis=0)
        upper_hsv = np.percentile(selected_pixels, 95, axis=0)

        lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        return lower_hsv, upper_hsv
    