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

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_test, range(150)))

    true_labels = []
    predicted_labels = []

    for true, pred in results:
        true_labels.extend(true)
        predicted_labels.extend(pred)

    correct_sum = sum(t == p for t, p in zip(true_labels, predicted_labels))
    incorrect_sum = len(true_labels) - correct_sum

    print(correct_sum, incorrect_sum)
    print(correct_sum / (correct_sum + incorrect_sum) * 100)

    plot_confusion_matrix(true_labels, predicted_labels)
