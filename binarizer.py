import cv2
import numpy as np

class Binerizer:
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
    


## test zakresow
image = cv2.imread(r"original_images\Y_P_hgr1_id12_3.jpg")
mask = cv2.imread(r"masks\Y_P_hgr1_id12_3.bmp", cv2.IMREAD_GRAYSCALE)

cv2.imshow("image", image)
cv2.imshow("mask", mask)
binarizer = HSVColorBinarizer()

lower_hsv, upper_hsv = binarizer.get_range_from_mask(image, mask)

print("Lower HSV:", lower_hsv)
print("Upper HSV:", upper_hsv)

## test drugiego - wybor maski
import cv2
image = cv2.imread(r"original_images\Y_P_hgr1_id12_3.jpg")

binarizer = HSVColorBinarizer(image)

mask = binarizer.get_mask_from_range(lower_hsv, upper_hsv)

cv2.imshow("Maska", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
#    def get_range_from_mask(self, image_bgr, mask, hand_is_black=True, sigma=3):
#        if image_bgr is None:
#            raise ValueError("Obraz nie może być None.")
#
#        if mask is None:
#            raise ValueError("Maska nie może być None.")
#
#        if image_bgr.shape[:2] != mask.shape[:2]:
#            raise ValueError(
#                f"Obraz i maska mają różne rozmiary: "
#                f"image={image_bgr.shape[:2]}, mask={mask.shape[:2]}"
#            )
#
#        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
#
#        if hand_is_black:
#            hand_area = mask == 0
#        else:
#            hand_area = mask == 255
#
#        selected_pixels = hsv[hand_area]
#
#        if selected_pixels.size == 0:
#            raise ValueError("Maska nie zawiera żadnych pikseli dłoni.")
#
#        selected_pixels = selected_pixels.astype(np.float32)
#
#        mean_hsv = selected_pixels.mean(axis=0)
#        std_hsv = selected_pixels.std(axis=0)
#
#        lower_sigma = mean_hsv - sigma * std_hsv
#        upper_sigma = mean_hsv + sigma * std_hsv
#
#        lower_sigma = np.maximum(lower_sigma, [0, 0, 0])
#        upper_sigma = np.minimum(upper_sigma, [179, 255, 255])
#
#        # Odrzucenie pikseli spoza przedziału 3 sigma
#        pixels_in_range = np.all(
#            (selected_pixels >= lower_sigma) & (selected_pixels <= upper_sigma),
#            axis=1
#        )
#
#        filtered_pixels = selected_pixels[pixels_in_range]
#
#        if filtered_pixels.size == 0:
#            raise ValueError("Po odrzuceniu pikseli spoza 3 sigma nie zostały żadne piksele.")
#
#        # Finalny zakres HSV liczony już tylko z pikseli po filtracji 3 sigma
#        lower_hsv = filtered_pixels.min(axis=0)
#        upper_hsv = filtered_pixels.max(axis=0)
#
#        lower_hsv = np.maximum(lower_hsv, [0, 0, 0])
#        upper_hsv = np.minimum(upper_hsv, [179, 255, 255])
#
#        return lower_hsv.astype(np.uint8), upper_hsv.astype(np.uint8)