from feature_extractor import FEATURE_NAMES, FeatureExtractor
from presenter import Presenter
from binarizer import Binarizer
import random


class Model:
    def __init__(self, image_db, test_percentage=0.1):
        self.feature_database = [[], []]

        images, masks, labels = image_db
        db_labeled = {}
        binarizer = Binarizer()
        self.lower_hsv, self.upper_hsv = binarizer.get_range_from_mask(
            images[0], masks[0]
        )
        for i in range(len(images)):
            lower_hsv, upper_hsv = binarizer.get_range_from_mask(images[i], masks[i])
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
            extractor = FeatureExtractor(model_masks[i])
            vis, features = extractor.process()

            self.feature_database[0].append(model_labels[i])
            self.feature_database[1].append(features)

    def _dist(self, featuresA, featuresB):
        sqr_sum = 0
        for i in range(len(featuresA)):
            sqr_sum += (featuresA[i] - featuresB[i]) ** 2

        return sqr_sum

    def Classify(self, image):
        extractor = FeatureExtractor(image)
        vis, features = extractor.process()

        closest = 0
        min_dist = self._dist(features, self.feature_database[1][closest])

        for i in range(len(self.feature_database[1])):
            dist = self._dist(features, self.feature_database[1][i])

            if dist < min_dist:
                closest = i
                min_dist = dist

        predicted_label = self.feature_database[0][closest]

        return vis, features, predicted_label

    def Test(self):
        images, masks, labels = self.test_database

        correct_count = 0
        incorrect_count = 0

        binarizer = Binarizer()
        for i in range(len(images)):
            mask = binarizer.get_mask_from_range(
                self.lower_hsv, self.upper_hsv, images[i]
            )
            vis, features, predicted_label = self.Classify(mask)
            vis["created_mask"] = mask
            vis["mask"] = masks[i]

            true_label = labels[i]

            print("=" * 60)
            print(f"TEST SAMPLE: {i}")
            print(f"TRUE LABEL     : {true_label}")
            print(f"PREDICTED LABEL: {predicted_label}")
            print("FEATURES:")
            presenter = Presenter()
            presenter.show(predicted_label, vis, 2)

            for name, value in zip(FEATURE_NAMES, features):
                print(f"{name:22s}: {value}")

            correct_count += predicted_label == true_label
            incorrect_count += predicted_label != true_label

        return correct_count, incorrect_count
