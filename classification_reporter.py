import numpy as np
import matplotlib.pyplot as plt

from binarizer import Binarizer
from feature_extractor import FEATURE_NAMES


class ClassificationReporter:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.features = []

    def add_iteration(self, model):
        images, masks, labels = model.test_database
        binarizer = Binarizer()

        for i in range(len(images)):
            created_mask = binarizer.get_mask_from_range(
                model.lower_hsv,
                model.upper_hsv,
                images[i]
            )

            _, features, predicted_label = model.Classify(created_mask)

            self.y_true.append(labels[i])
            self.y_pred.append(predicted_label)
            self.features.append(features)

    def run(self):
        print("Liczba wszystkich próbek testowych:", len(self.y_true))

        correct = sum(t == p for t, p in zip(self.y_true, self.y_pred))
        incorrect = len(self.y_true) - correct

        print("Poprawne:", correct)
        print("Błędne:", incorrect)
        print("Accuracy:", correct / len(self.y_true) * 100)

        self._plot_confusion_matrix()

    def _plot_confusion_matrix(self):
        labels = sorted(set(self.y_true) | set(self.y_pred))
        label_to_idx = {label: i for i, label in enumerate(labels)}

        matrix = np.zeros((len(labels), len(labels)), dtype=int)

        for true_label, predicted_label in zip(self.y_true, self.y_pred):
            row = label_to_idx[true_label]
            col = label_to_idx[predicted_label]
            matrix[row, col] += 1

        print("Macierz błędów:")
        print(matrix)
        print("Suma macierzy:", matrix.sum())

        plt.figure(figsize=(8, 6))
        plt.imshow(matrix)

        plt.title(f"Macierz błędów - wszystkie iteracje, n={matrix.sum()}")
        plt.xlabel("Przewidziana klasa")
        plt.ylabel("Prawdziwa klasa")

        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, matrix[i, j], ha="center", va="center")

        plt.colorbar()
        plt.tight_layout()
        plt.show()