import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_visualization(subtitle, results, width=4):
    names = list(results.keys())
    images = list(results.values())
    n = len(images)

    if n == 0:
        raise ValueError("results is empty")

    rows = int(np.ceil(n / width))

    fig, axes = plt.subplots(rows, width, figsize=(4 * width, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * width):
        ax = axes[i]

        if i >= n:
            ax.axis("off")
            continue

        img = images[i]
        name = names[i]

        if img is None:
            ax.set_title(f"{name} (None)")
            ax.axis("off")
            continue

        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(str(name))
        ax.axis("off")

    fig.suptitle(subtitle, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true, pred):
    labels = sorted(set(true) | set(pred))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(true, pred):
        row = label_to_idx[true_label]
        col = label_to_idx[predicted_label]
        matrix[row, col] += 1

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
