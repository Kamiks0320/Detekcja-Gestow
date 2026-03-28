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

    def process(self):
        image = self.load_image()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 48, 80], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape, dtype=np.uint8)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        cv2.drawContours(mask, largest_contours, -1, 255, thickness=cv2.FILLED)

        hull = mask.copy()
        cv2.drawContours(hull, [cv2.convexHull(largest_contours[0])], -1, 128, 2)

        return {
            "original": image,
            "mask": mask,
            "hull": hull,
        }


def main():
    path = "original_images\\"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random_files = random.sample(files, 20)

    image_scale = 0.5
    results_scale = 1
    resulting_images = {}
    for image_path in random_files:
        processor = HandPreprocessingFromImage(
            image_path=path + image_path, output_size=256
        )

        try:
            results = processor.process()
        except Exception as e:
            print(f"Błąd: {e}")
            return

        for im in results:

            def resize_to_height(img, height):
                ratio = height / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * ratio), height))

            h = 100  # desired height
            resized = resize_to_height(results[im], h)
            if im in resulting_images:
                resulting_images[im] = np.hstack((resulting_images[im], resized))
            else:
                resulting_images[im] = resized

    for im in resulting_images:
        cv2.imshow(
            im,
            cv2.resize(
                resulting_images[im],
                None,
                fx=results_scale,
                fy=results_scale,
                interpolation=cv2.INTER_NEAREST,
            ),
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
