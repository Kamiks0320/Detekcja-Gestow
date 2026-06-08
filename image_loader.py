from pathlib import Path
import random
import cv2


class ImageLoader:
    def __init__(self, image_root="experiment_6", mask_root="experiment_7"):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root) if mask_root is not None else None

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_root}"
            )

        if self.mask_root is not None and not self.mask_root.exists():
            raise FileNotFoundError(
                f"Mask directory does not exist: {self.mask_root}"
            )

    def _get_image_files(self):
        allowed_extensions = {".jpg", ".jpeg", ".png"}

        image_files = []
        seen_paths = set()

        for path in self.image_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                normalized_path = str(path.resolve()).lower()

                if normalized_path not in seen_paths:
                    seen_paths.add(normalized_path)
                    image_files.append(path)

        image_files = sorted(image_files)

        if len(image_files) == 0:
            raise FileNotFoundError(f"No image files found in: {self.image_root}")

        return image_files

    def _get_mask_files(self):
        if self.mask_root is None:
            raise ValueError("mask_root is None, cannot load masks.")

        allowed_extensions = {".bmp", ".png", ".jpg", ".jpeg"}

        mask_files = []
        seen_paths = set()

        for path in self.mask_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                normalized_path = str(path.resolve()).lower()

                if normalized_path not in seen_paths:
                    seen_paths.add(normalized_path)
                    mask_files.append(path)

        mask_files = sorted(mask_files)

        if len(mask_files) == 0:
            raise FileNotFoundError(f"No mask files found in: {self.mask_root}")

        return mask_files

    def load_images_only(self):
        image_files = self._get_image_files()

        images = []
        file_names = []
        labels = []

        for image_path in image_files:
            images.append(self._load_image(image_path, grayscale=False))
            file_names.append(image_path.name)
            labels.append(image_path.name[0])

        return images, file_names, labels

    def load_all(self):
        if self.mask_root is None:
            raise ValueError("mask_root is None, use load_images_only() instead.")

        image_files = self._get_image_files()

        images = []
        masks = []
        file_names = []
        labels = []

        for image_path in image_files:
            relative_path = image_path.relative_to(self.image_root)

            mask_path = (self.mask_root / relative_path).with_suffix(".bmp")

            if not mask_path.exists():
                # fallback, gdy maski są płasko w folderze masks_cropped
                mask_path = (self.mask_root / image_path.name).with_suffix(".bmp")

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image {image_path.name}: {mask_path}"
                )

            images.append(self._load_image(image_path, grayscale=False))
            masks.append(self._load_image(mask_path, grayscale=True))
            file_names.append(image_path.name)
            labels.append(image_path.name[0])

        return images, masks, file_names, labels

    def load_images(self, names):
        if not isinstance(names, (list, tuple)):
            raise TypeError("names must be a list or tuple")

        images = []
        labels = []

        for name in names:
            path = self.image_root / name
            images.append(self._load_image(path, grayscale=False))
            labels.append(Path(name).name[0])

        return images, labels

    def load_masks(self, names):
        if self.mask_root is None:
            raise ValueError("mask_root is None, cannot load masks.")

        if not isinstance(names, (list, tuple)):
            raise TypeError("names must be a list or tuple")

        masks = []
        labels = []

        for name in names:
            path = self.mask_root / name
            if not path.exists():
                path = (self.mask_root / name).with_suffix(".bmp")

            masks.append(self._load_image(path, grayscale=True))
            labels.append(Path(name).name[0])

        return masks, labels

    def load_images_random(self, n):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n <= 0:
            raise ValueError("n must be greater than 0")

        files = self._get_image_files()
        sample_n = min(n, len(files))
        random_files = random.sample(files, sample_n)

        images = []
        file_names = []
        labels = []

        for path in random_files:
            images.append(self._load_image(path, grayscale=False))
            file_names.append(path.name)
            labels.append(path.name[0])

        return images, file_names, labels

    def load_masks_random(self, n):
        if self.mask_root is None:
            raise ValueError("mask_root is None, cannot load masks.")

        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n <= 0:
            raise ValueError("n must be greater than 0")

        files = self._get_mask_files()
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

    def _load_image(self, image_path, grayscale=True):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"File does not exist: {image_path}")

        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(image_path), flag)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return img