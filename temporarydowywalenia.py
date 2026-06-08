from image_loader import ImageLoader
from feature_extractor import extract_features

from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import random
import time


# =========================
# KONFIGURACJA
# =========================

REPEATS = 150
TEST_PERCENTAGE = 0.1
USE_PROCESS_POOL = True
# Folder, w którym są gotowe maski z bazy
MASK_ROOT = "experiment_4"

GLOBAL_IMAGES = None
GLOBAL_MASKS = None
GLOBAL_LABELS = None


# =========================
# MODEL - EKSPERYMENT 2
# Używa gotowych masek z bazy
# Nie tworzy masek HSV
# =========================

class ModelDatabaseMasks:
    def __init__(self, image_db, test_percentage=0.1):
        train_start = time.perf_counter()

        self.feature_database = [[], []]

        images, masks, labels = image_db

        db_labeled = {}

        for i in range(len(images)):
            label = labels[i]

            if label not in db_labeled:
                db_labeled[label] = []

            db_labeled[label].append(i)

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

        # Trening:
        # cechy są liczone bezpośrednio z gotowych masek z bazy
        for i in range(len(model_masks)):
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

    def classify(self, mask):
        # Predykcja:
        # NIE robimy get_mask_from_range()
        # NIE używamy HSV
        # Bierzemy gotową maskę z bazy
        vis, features = extract_features(mask)

        distances = []

        for label, feature_vector in zip(
            self.feature_database[0],
            self.feature_database[1]
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

    def test(self):
        prediction_start = time.perf_counter()

        true_labels = []
        predicted_labels = []

        for image, mask, label in zip(*self.test_database):
            vis, features, prediction = self.classify(mask)

            true_labels.append(label)
            predicted_labels.append(prediction)

        prediction_end = time.perf_counter()
        prediction_time = prediction_end - prediction_start

        return true_labels, predicted_labels, self.training_time, prediction_time


# =========================
# MACIERZ POMYŁEK
# =========================

def plot_confusion_matrix(true_labels, predicted_labels):
    labels = sorted(set(true_labels) | set(predicted_labels))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        row = label_to_idx[true_label]
        col = label_to_idx[predicted_label]
        matrix[row, col] += 1

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)

    plt.title(f"Macierz pomyłek - eksperyment 2, n={matrix.sum()}")
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Prawdziwa klasa")

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("experiment_2_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# FUNKCJE DO EKSPERYMENTU
# =========================

def init_worker(images, masks, labels):
    global GLOBAL_IMAGES
    global GLOBAL_MASKS
    global GLOBAL_LABELS

    GLOBAL_IMAGES = images
    GLOBAL_MASKS = masks
    GLOBAL_LABELS = labels


def run_test(iteration):
    random.seed(42 + iteration)

    model = ModelDatabaseMasks(
        image_db=(GLOBAL_IMAGES, GLOBAL_MASKS, GLOBAL_LABELS),
        test_percentage=TEST_PERCENTAGE
    )

    return model.test()


def calculate_iteration_accuracy(true_labels, predicted_labels):
    correct = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )

    incorrect = len(true_labels) - correct

    if correct + incorrect == 0:
        return 0

    return correct / (correct + incorrect) * 100


# =========================
# MAIN
# =========================

def main():
    global GLOBAL_IMAGES
    global GLOBAL_MASKS
    global GLOBAL_LABELS

    print("Ładowanie masek z bazy...")

    # Nie używamy load_all(), bo ono szuka obrazów .jpg.
    # W eksperymencie 2 potrzebujemy tylko masek .bmp.
    loader = ImageLoader(
        image_root=MASK_ROOT,
        mask_root=MASK_ROOT
    )

    # Bierzemy bardzo dużą liczbę, a loader i tak weźmie maksymalnie tyle,
    # ile faktycznie jest masek w folderze.
    masks, file_names, labels = loader.load_masks_random(10**9)

    # Model ma jeszcze parametr images, ale w eksperymencie 2 obrazy RGB nie są używane.
    # Dlatego tworzymy pustą listę tej samej długości.
    images = [None] * len(masks)

    GLOBAL_IMAGES = images
    GLOBAL_MASKS = masks
    GLOBAL_LABELS = labels

    print(f"Folder masek: {MASK_ROOT}")
    print(f"Liczba masek: {len(masks)}")
    print(f"Liczba etykiet: {len(labels)}")
    print(f"Klasy: {sorted(set(labels))}")

    print("\nEksperyment 2: predykcja na maskach z bazy")
    print("=" * 60)
    print(f"Liczba iteracji: {REPEATS}")
    print(f"Procent testowy: {TEST_PERCENTAGE * 100:.0f}%")
    print("Segmentacja HSV: NIE")
    print("Źródło maski: gotowa maska z bazy")

    experiment_start = time.perf_counter()

    if USE_PROCESS_POOL:
        with ProcessPoolExecutor(
            initializer=init_worker,
            initargs=(images, masks, labels)
        ) as executor:
            results = list(executor.map(run_test, range(REPEATS)))
    else:
        results = []

        for i in range(REPEATS):
            print(f"Iteracja {i + 1}/{REPEATS}")
            results.append(run_test(i))

    experiment_end = time.perf_counter()
    real_experiment_time = experiment_end - experiment_start

    true_labels_all = []
    predicted_labels_all = []

    training_times = []
    prediction_times = []
    iteration_accuracies = []

    for true_labels, predicted_labels, training_time, prediction_time in results:
        true_labels_all.extend(true_labels)
        predicted_labels_all.extend(predicted_labels)

        training_times.append(training_time)
        prediction_times.append(prediction_time)

        iteration_accuracy = calculate_iteration_accuracy(true_labels, predicted_labels)
        iteration_accuracies.append(iteration_accuracy)

    correct_sum = sum(
        true_label == predicted_label
        for true_label, predicted_label in zip(true_labels_all, predicted_labels_all)
    )

    incorrect_sum = len(true_labels_all) - correct_sum

    accuracy = correct_sum / (correct_sum + incorrect_sum) * 100

    total_training_time = sum(training_times)
    total_prediction_time = sum(prediction_times)

    avg_training_time = total_training_time / len(training_times)
    avg_prediction_time = total_prediction_time / len(prediction_times)

    total_predictions = len(true_labels_all)
    avg_prediction_per_image = total_prediction_time / total_predictions

    print("\n" + "=" * 60)
    print("PODSUMOWANIE EKSPERYMENTU 2")
    print("=" * 60)

    print("\nWyniki:")
    print(f"Liczba wszystkich predykcji: {total_predictions}")
    print(f"Poprawne klasyfikacje: {correct_sum}")
    print(f"Błędne klasyfikacje: {incorrect_sum}")
    print(f"Accuracy globalne: {accuracy:.2f}%")

    print("\nAccuracy z iteracji:")
    print(f"Średnie accuracy z iteracji: {np.mean(iteration_accuracies):.2f}%")
    print(f"Najniższe accuracy: {np.min(iteration_accuracies):.2f}%")
    print(f"Najwyższe accuracy: {np.max(iteration_accuracies):.2f}%")
    print(f"Odchylenie standardowe accuracy: {np.std(iteration_accuracies):.2f}")

    print("\nCzasy liczone jako suma czasów z iteracji:")
    print(f"Suma czasu treningu: {total_training_time:.4f} s")
    print(f"Suma czasu predykcji: {total_prediction_time:.4f} s")
    print(f"Średni czas treningu jednej iteracji: {avg_training_time:.4f} s")
    print(f"Średni czas predykcji jednej iteracji: {avg_prediction_time:.4f} s")
    print(f"Średni czas predykcji jednego obrazu: {avg_prediction_per_image:.6f} s")

    print("\nRzeczywisty czas wykonania programu:")
    print(f"Czas rzeczywisty: {real_experiment_time:.4f} s")

    plot_confusion_matrix(true_labels_all, predicted_labels_all)

    print("\nMacierz pomyłek zapisana do:")
    print("experiment_2_confusion_matrix.png")


if __name__ == "__main__":
    main()