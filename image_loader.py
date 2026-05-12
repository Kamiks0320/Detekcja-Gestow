from pathlib import Path
import random
import cv2


class ImageLoader:
    def __init__(self, image_root="original_cropped", mask_root="masks_cropped"):
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
        labels = []

        for name in names:
            path = self.mask_root / name
            masks.append(self._load_image(path, grayscale=True))
            labels.append(name[0])

        return masks, labels

    def load_masks_random(self, n):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n <= 0:
            raise ValueError("n must be greater than 0")

        ext = "*.bmp"
        files = list(self.mask_root.glob(f"**/{ext}"))

        if len(files) == 0:
            raise FileNotFoundError(f"No mask files found in: {self.mask_root}")
        files = sorted(files)

        sample_n = min(n, len(files))
        random_files = random.sample(files, sample_n)

        masks = []
        file_names = []
        labels = []

        for path in random_files:
            masks.append(self._load_image(path, grayscale=True))
            file_names.append(path.name)
            labels.append(path.name[0])

        return masks, file_names, labels

    def load_images(self, names):
        if not isinstance(names, (list, tuple)):
            raise TypeError("names must be a list or tuple")

        images = []
        labels = []

        for name in names:
            path = self.image_root / name
            images.append(self._load_image(path, grayscale=False))
            labels.append(name[0])

        return images, labels

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

        files = sorted(files)
        sample_n = min(n, len(files))
        random_files = random.sample(files, sample_n)

        images = []
        file_names = []
        labels = []

        for path in random_files:
            images.append(self._load_image(path, grayscale=False))
            file_names.append(path.name)
            labels.append(path.name[0])

        return masks, file_names, labels

    def load_all(self):
        extension = "*.jpg"
        image_files = []
        image_files.extend(self.image_root.glob(f"**/{extension}"))

        if len(image_files) == 0:
            raise FileNotFoundError(f"No image files found in: {self.image_root}")
        image_files = sorted(image_files)

        images = []
        masks = []
        file_names = []
        labels = []

        for image_path in image_files:

            mask_path = (self.mask_root / image_path.name).with_suffix(".bmp")
            if not mask_path.exists():
                raise FileNotFoundError(f"Image not found: {mask_path}")

            images.append(self._load_image(image_path, grayscale=False))
            masks.append(self._load_image(mask_path, grayscale=True))
            file_names.append(image_path.name)
            labels.append(image_path.name[0])

        return images, masks, file_names, labels

    def _load_image(self, image_path, grayscale=True):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(image_path), flag)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return img
