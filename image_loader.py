from pathlib import Path
import random
import cv2


class ImageLoader:
    def __init__(self, image_root="original_images", mask_root="masks_cropped"):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_root}"
            )

        if not self.mask_root.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_root}")

    def load_masks(self, names):
        if not isinstance(names, (list, tuple)):
            raise TypeError("names must be a list or tuple")

        masks = []

        for name in names:
            path = self.mask_root / name
            masks.append(self._load_image(path, grayscale=True))

        return masks

    def load_masks_random(self, n):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n <= 0:
            raise ValueError("n must be greater than 0")

        ext = "*.bmp"
        files = list(self.mask_root.glob(f"**/{ext}"))

        if len(files) == 0:
            raise FileNotFoundError(f"No mask files found in: {self.mask_root}")

        sample_n = min(n, len(files))
        random_files = random.sample(files, sample_n)

        masks = []
        file_names = []

        for path in random_files:
            masks.append(self._load_image(path, grayscale=True))
            file_names.append(path.name)

        return masks, file_names

    def load_images(self, names):
        if not isinstance(names, (list, tuple)):
            raise TypeError("names must be a list or tuple")

        images = []

        for name in names:
            path = self.image_root / name
            images.append(self._load_image(path, grayscale=False))

        return images

    def load_images_random(self, n):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n <= 0:
            raise ValueError("n must be greater than 0")

        extensions = ["*.jpg", "*.JPG"]
        files = []
        for ext in extensions:
            files.extend(self.image_root.glob(f"**/{ext}"))

        if len(files) == 0:
            raise FileNotFoundError(f"No image files found in: {self.image_root}")

        sample_n = min(n, len(files))
        random_files = random.sample(files, sample_n)

        images = []
        file_names = []

        for path in random_files:
            images.append(self._load_image(path, grayscale=False))
            file_names.append(path.name)

        return masks, file_names

    def _load_image(self, image_path, grayscale=True):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(image_path), flag)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return img
