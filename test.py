import cv2
import numpy as np


class HandPreprocessingFromImage:
    def __init__(self, image_path, output_size=256):
        self.image_path = image_path
        self.output_size = output_size

        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)

    def load_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Nie udało się wczytać obrazu: {self.image_path}")
        return image

    def skin_segmentation(self, image):
        """
        Segmentacja skóry na podstawie koloru:
        - YCrCb
        - HSV
        - połączenie masek
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Zakresy skóry w YCrCb
        lower_ycrcb = np.array([0, 145, 90], dtype=np.uint8)
        upper_ycrcb = np.array([255, 170, 120], dtype=np.uint8)

        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Zakresy skóry w HSV

        lower_hsv = np.array([0, 15, 70], dtype=np.uint8)
        upper_hsv = np.array([17, 170, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Połączenie obu masek
        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        return mask_ycrcb, mask_hsv, skin_mask

    def clean_mask(self, mask):
        """
        Czyszczenie maski:
        - median blur
        - open
        - close
        """
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        return mask

    def extract_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) > 5000:  # zwiększ próg
                cv2.drawContours(filtered, [c], -1, 255, -1)

        mask = filtered

        if not contours:
            return None, np.zeros_like(mask)

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 1000:
            return None, np.zeros_like(mask)

        hand_mask = np.zeros_like(mask)
        cv2.drawContours(hand_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Jeszcze jedno domknięcie po narysowaniu obiektu
        hand_mask = cv2.morphologyEx(
            hand_mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2
        )

        return largest, hand_mask

    def normalize_hand(self, image, hand_mask, contour):
        if contour is None:
            empty_img = np.zeros(
                (self.output_size, self.output_size, 3), dtype=np.uint8
            )
            empty_mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
            return empty_img, empty_mask

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
        scale = min(self.output_size / crop_w, self.output_size / crop_h)

        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))

        resized_img = cv2.resize(
            hand_only, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        resized_mask = cv2.resize(
            mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        canvas_img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        canvas_mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)

        offset_x = (self.output_size - new_w) // 2
        offset_y = (self.output_size - new_h) // 2

        canvas_img[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = (
            resized_img
        )
        canvas_mask[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = (
            resized_mask
        )

        return canvas_img, canvas_mask

    def process(self):
        image = self.load_image()

        mask_ycrcb, mask_hsv, raw_skin_mask = self.skin_segmentation(image)
        cleaned_mask = self.clean_mask(raw_skin_mask)
        contour, hand_mask = self.extract_largest_contour(cleaned_mask)
        normalized_hand, normalized_mask = self.normalize_hand(
            image, hand_mask, contour
        )

        vis = image.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return {
            "original": image,
            "mask_ycrcb": mask_ycrcb,
            "mask_hsv": mask_hsv,
            "raw_skin_mask": raw_skin_mask,
            "cleaned_mask": cleaned_mask,
            "hand_mask": hand_mask,
            "contour_view": vis,
            "normalized_hand": normalized_hand,
            "normalized_mask": normalized_mask,
        }


def main():
    image_path = r"cropped\1_P_hgr1_id04_1.jpg"

    processor = HandPreprocessingFromImage(image_path=image_path, output_size=256)

    try:
        results = processor.process()
    except Exception as e:
        print(f"Błąd: {e}")
        return

    cv2.imshow("1. Original", results["original"])
    cv2.imshow("2. Skin mask YCrCb", results["mask_ycrcb"])
    cv2.imshow("3. Skin mask HSV", results["mask_hsv"])
    cv2.imshow("4. Combined raw skin mask", results["raw_skin_mask"])
    cv2.imshow("5. Cleaned mask", results["cleaned_mask"])
    cv2.imshow("6. Hand mask", results["hand_mask"])
    cv2.imshow("7. Contour + bounding box", results["contour_view"])
    cv2.imshow("8. Normalized hand", results["normalized_hand"])
    cv2.imshow("9. Normalized mask", results["normalized_mask"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

