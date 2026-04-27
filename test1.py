import cv2
import numpy as np
import random
import os


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

    def get_mask_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 48, 80], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        mid = (lower.astype(np.float32) + upper.astype(np.float32)) / 2
        hsv_f = hsv.astype(np.float32)
        dist = 1 / np.sqrt(
            (hsv_f[:, :, 0] - mid[0]) ** 2
            + (hsv_f[:, :, 1] - mid[1]) ** 2
            + (hsv_f[:, :, 2] - mid[2]) ** 2
        )
        dist_gray = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_gray = cv2.medianBlur(dist_gray, 5)
        return dist_gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape, dtype=np.uint8)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        cv2.drawContours(mask, largest_contours, -1, 255, thickness=cv2.FILLED)
        return mask

    def get_mask_edge(self, image):
        # mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(image, (3, 3), 0)
        mask = cv2.Canny(image, 1, 60)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape, dtype=np.uint8)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        cv2.drawContours(mask, largest_contours, -1, 255, thickness=cv2.FILLED)

    def process(self):
        image = self.load_image()

        mask_color = self.get_mask_color(image)
        mask_edge = cv2.addWeighted(
            self.get_mask_edge(mask_color), 0.8, self.get_mask_edge(image), 0.2, 0
        )
        mask_sum = cv2.addWeighted(mask_edge, 0.4, mask_color, 0.6, 0)

        _, mask = cv2.threshold(mask_sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_color_bin = cv2.threshold(
            mask_color, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, mask_edge_bin = cv2.threshold(
            mask_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return {
            "original": image,
            "mask_color": mask_color,
            "mask_color_bin": mask_color_bin,
            "mask_edge_bin": mask_edge_bin,
            "mask_edge": mask_edge,
            "mask_sum": mask_sum,
            "mask": mask,
        }


def main():
    #path = "original_images\\"
    path = os.path.join(".", "original_images")
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random_files = random.sample(files, 1)

    image_size = 400
    for image_path in random_files:
        full_path = os.path.join(path, image_path)
        # image_path = "O_P_hgr1_id07_2.jpg"
        processor = HandPreprocessingFromImage(
            image_path=full_path, output_size=256
        )

        try:
            results = processor.process()
        except Exception as e:
            print(f"Błąd: {e}")
            return

        print(image_path)
        for im in results:
            height, width = results[im].shape[:2]
            cv2.imshow(
                im,
                cv2.resize(
                    results[im],
                    None,
                    fx=image_size / width,
                    fy=image_size / height,
                    interpolation=cv2.INTER_NEAREST,
                ),
            )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
