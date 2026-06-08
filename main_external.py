from image_loader import ImageLoader
from presenter import plot_confusion_matrix
from model import Model
import time
import statistics


REPEATS = 10


train_loader = ImageLoader(
    image_root="original_cropped",
    mask_root="masks_cropped"
)

train_images, train_masks, train_file_names, train_labels = train_loader.load_all()


external_loader = ImageLoader(
    image_root="experiment_6",
    mask_root=None
)

external_images, external_file_names, external_labels = external_loader.load_images_only()


print("DANE TRENINGOWE Z BAZY:")
print(f"Liczba obrazów treningowych: {len(train_images)}")
print(f"Przykładowy plik treningowy: {train_file_names[0]}")

print()
print("DANE TESTOWE KOLEGI:")
print(f"Liczba obrazów kolegi: {len(external_images)}")
print(f"Przykładowy plik kolegi: {external_file_names[0]}")

accuracies = []
correct_counts = []
incorrect_counts = []
training_times = []
prediction_times = []
prediction_per_image_times = []

all_true_labels = []
all_predicted_labels = []

last_predicted_labels = []


for i in range(REPEATS):
    print(f"Iteracja {i + 1}/{REPEATS}")

    model = Model(
        (train_images, train_masks, train_labels),
        test_percentage=0.0
    )

    prediction_start = time.perf_counter()

    predicted_labels = model.predict_external(
        external_images,
        file_names=external_file_names,
        save_masks=True,
        output_dir="created_masks_external"
    )

    prediction_time = time.perf_counter() - prediction_start

    correct_sum = sum(
        true == pred for true, pred in zip(external_labels, predicted_labels)
    )

    incorrect_sum = len(external_labels) - correct_sum
    accuracy = correct_sum / len(external_labels) * 100

    accuracies.append(accuracy)
    correct_counts.append(correct_sum)
    incorrect_counts.append(incorrect_sum)
    training_times.append(model.training_time)
    prediction_times.append(prediction_time)
    prediction_per_image_times.append(prediction_time / len(external_images))

    all_true_labels.extend(external_labels)
    all_predicted_labels.extend(predicted_labels)

    last_predicted_labels = predicted_labels


print()
print("Predykcje z ostatniej iteracji, tylko zdjęcia kolegi:")
for file_name, true_label, predicted_label in zip(
    external_file_names,
    external_labels,
    last_predicted_labels
):
    print(f"{file_name}: prawdziwa={true_label}, przewidziana={predicted_label}")


print()
print(f"Wyniki średnie z {REPEATS} iteracji:")
print(f"Średnia liczba poprawnych: {statistics.mean(correct_counts):.2f}")
print(f"Średnia liczba błędnych: {statistics.mean(incorrect_counts):.2f}")
print(f"Średnie accuracy: {statistics.mean(accuracies):.2f}%")

if len(accuracies) > 1:
    print(f"Odchylenie standardowe accuracy: {statistics.stdev(accuracies):.4f}")


print()
print(f"Czasy średnie z {REPEATS} iteracji:")
print(f"Suma czasu treningu: {sum(training_times):.4f} s")
print(f"Suma czasu predykcji: {sum(prediction_times):.4f} s")
print(f"Średni czas treningu jednej iteracji: {statistics.mean(training_times):.4f} s")
print(f"Średni czas predykcji jednej iteracji: {statistics.mean(prediction_times):.4f} s")
print(f"Średni czas predykcji jednego obrazu: {statistics.mean(prediction_per_image_times):.6f} s")


plot_confusion_matrix(all_true_labels, all_predicted_labels)