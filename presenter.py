import matplotlib.pyplot as plt
import cv2
import numpy as np
import uuid
import os

# Funkcja wyswietlajaca wizualizacje obrazow w formie siatki.
# 
# Postac wywolanania:
#       show_visualization(subtitle="Wizualizacja", results={"Obraz 1": image1, "Obraz 2": image2}, width=2, save=False)
# 
# subtitle - napis wyswietlany nad wizualizacja.
# results - slownik, gdzie klucze to napisy z nazwami obrazow, a wartosci to obrazy w formacie BGR (np. wczytane za pomoca cv2.imread) lub obrazy w formacie grayscale (2D).
# width - liczba obrazow w jednym wierszu wizualizacji. Domyslnie 4.
# save - boolean okreslajacy, czy zapisac wizualizacje do pliku zamiast wyswietlac. Domyslnie False (wyswietla wizualizacje). Jesli True, wizualizacja zostanie zapisana do katalogu "failed" z unikalna nazwa pliku.
def show_visualization(subtitle, results, width=4, save=False):
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
    if not save:
        plt.show()
    else:
        filename = f"./failed/{uuid.uuid4().hex}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

# Funkcja plot_confusion_matrix tworzy i wyswietla macierz bledow na podstawie listy prawdziwych etykiet i przewidywanych etykiet.
#
# Postac wywolanania:
#       plot_confusion_matrix(true_labels, predicted_labels)
# 
# true_labels - lista prawdziwych etykiet dla danych testowych.
# predicted_labels - lista przewidywanych etykiet dla danych testowych. Powinna byc tej samej dlugosci co true_labels. Etykiety powinny byc pojedynczymi znakami reprezentujacymi gest dloni (np. "1", "L", "O").
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
