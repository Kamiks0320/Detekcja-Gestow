from image_loader import ImageLoader
from presenter import show_visualization, plot_confusion_matrix
from model import Model
from concurrent.futures import ProcessPoolExecutor

loader = ImageLoader()
images, masks, file_names, labels = loader.load_all()


def run_test(_):
    model = Model((images, masks, labels), test_percentage=0.1)
    return model.Test()


if __name__ == "__main__":
    training_times = []
    prediction_times = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_test, range(150)))

    true_labels = []
    predicted_labels = []

    for true, pred, training_time, prediction_time in results:
        true_labels.extend(true)
        predicted_labels.extend(pred)

        training_times.append(training_time)
        prediction_times.append(prediction_time)

    correct_sum = sum(t == p for t, p in zip(true_labels, predicted_labels))
    incorrect_sum = len(true_labels) - correct_sum

    print(correct_sum, incorrect_sum)
    print(correct_sum / (correct_sum + incorrect_sum) * 100)


    total_training_time = sum(training_times)
    total_prediction_time = sum(prediction_times)
    
    avg_training_time = total_training_time / len(training_times)
    avg_prediction_time = total_prediction_time / len(prediction_times)

    total_predictions = len(true_labels)
    avg_prediction_per_image = total_prediction_time / total_predictions
    
    print("Czasy:")
    print(f"Suma czasu treningu: {total_training_time:.4f} s")
    print(f"Suma czasu predykcji: {total_prediction_time:.4f} s")
    print(f"Średni czas treningu jednej iteracji: {avg_training_time:.4f} s")
    print(f"Średni czas predykcji jednej iteracji: {avg_prediction_time:.4f} s")
    print(f"Średni czas predykcji jednego obrazu: {avg_prediction_per_image:.6f} s")

    plot_confusion_matrix(true_labels, predicted_labels)
