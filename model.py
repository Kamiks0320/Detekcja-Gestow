from feature_extractor import extract_features
from presenter import show_visualization
from binarizer import get_mask_from_range, get_range_from_mask
import random
import time
from pathlib import Path
import cv2

class Model:
    def __init__(self, image_db, test_percentage=0.1):
        train_start = time.perf_counter()

        self.feature_database = [[], []]

        images, masks, labels = image_db
        db_labeled = {}
        self.lower_hsv, self.upper_hsv = get_range_from_mask(images[0], masks[0])
        for i in range(len(images)):
            lower_hsv, upper_hsv = get_range_from_mask(images[i], masks[i])
            self.lower_hsv[0] = min(self.lower_hsv[0], lower_hsv[0])
            self.lower_hsv[1] = min(self.lower_hsv[1], lower_hsv[1])
            self.lower_hsv[2] = min(self.lower_hsv[2], lower_hsv[2])
            self.upper_hsv[0] = max(self.upper_hsv[0], upper_hsv[0])
            self.upper_hsv[1] = max(self.upper_hsv[1], upper_hsv[1])
            self.upper_hsv[2] = max(self.upper_hsv[2], upper_hsv[2])

            if labels[i] not in db_labeled:
                db_labeled[labels[i]] = []
            db_labeled[labels[i]].append(i)

        test_images = []
        test_masks = []
        test_labels = []
        model_images = []
        model_masks = []
        model_labels = []

        for label in db_labeled:
            indices = db_labeled[label].copy()
            random.shuffle(indices)

            if len(indices) < 2:
                raise ValueError(f"Za mało próbek dla klasy {label}. Minimum to 2.")

            if test_percentage <= 0:
                test_count = 0
            else:
                test_count = max(1, int(len(indices) * test_percentage))

                if test_count >= len(indices):
                    test_count = len(indices) - 1

            for j, idx in enumerate(indices):
                image = images[idx]
                mask = masks[idx]

                if j < test_count:
                    test_images.append(image)
                    test_masks.append(mask)
                    test_labels.append(label)
                else:
                    model_images.append(image)
                    model_masks.append(mask)
                    model_labels.append(label)

        self.test_database = test_images, test_masks, test_labels

        for i in range(len(model_images)):
            vis, features = extract_features(model_masks[i])

            self.feature_database[0].append(model_labels[i])
            self.feature_database[1].append(features)

        train_end = time.perf_counter()

        self.training_time = train_end - train_start

    def _dist(self, featuresA, featuresB):
        sqr_sum = 0
        for i in range(len(featuresA)):
            sqr_sum += (featuresA[i] - featuresB[i]) ** 2

        return sqr_sum
    def predict_external(self, images, file_names=None, save_masks=False, output_dir="created_masks_external"):
        predicted_labels = []

        if save_masks:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            vis, features, predicted_label = self.Classify(image)
            predicted_labels.append(predicted_label)

            if save_masks:
                mask = vis["created_mask"]

                if file_names is not None:
                    output_name = Path(file_names[i]).with_suffix(".bmp").name
                else:
                    output_name = f"mask_{i}.bmp"

                output_path = output_dir / output_name
                cv2.imwrite(str(output_path), mask)

        return predicted_labels
    def Classify(self, image):
            
        mask = get_mask_from_range(self.lower_hsv, self.upper_hsv, image)
        vis, features = extract_features(mask)
        vis["created_mask"] = mask

        distances = []

        for label, feature_vector in zip(
            self.feature_database[0], self.feature_database[1]
        ):
            dist = self._dist(features, feature_vector)
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        nearest = distances[:4]
        label_count = {}

        for _, label in nearest:
            label_count[label] = label_count.get(label, 0) + 1

        predicted_label = max(label_count, key=label_count.get)

        return vis, features, predicted_label
    def ClassifyMask(self, mask):
        vis, features = extract_features(mask)
        vis["created_mask"] = mask

        distances = []

        for label, feature_vector in zip(
            self.feature_database[0], self.feature_database[1]
        ):
            dist = self._dist(features, feature_vector)
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        nearest = distances[:4]
        label_count = {}

        for _, label in nearest:
            label_count[label] = label_count.get(label, 0) + 1

        predicted_label = max(label_count, key=label_count.get)

        return vis, features, predicted_label
    def predict_external_masks(self, masks):
        predicted_labels = []

        for mask in masks:
            vis, features, predicted_label = self.ClassifyMask(mask)
            predicted_labels.append(predicted_label)

        return predicted_labels
    def Test(self, visualize_errors=False):
        prediction_start = time.perf_counter()

        true_labels = []
        predicted_labels = []

        for image, mask, label in zip(*self.test_database):
            vis, features, prediction = self.Classify(image)
            vis["mask"] = mask

            true_labels.append(label)
            predicted_labels.append(prediction)

            if visualize_errors and label != prediction:
                show_visualization(label + " " + prediction, vis, 2)

        prediction_end = time.perf_counter()
        prediction_time = prediction_end - prediction_start

        return true_labels, predicted_labels, self.training_time, prediction_time
