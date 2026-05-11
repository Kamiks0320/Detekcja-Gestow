import matplotlib.pyplot as plt
import cv2
import numpy as np


class Presenter:
    # def __init__(self):
    def show(self, subtitle, results, width=4):
        if not isinstance(results, (dict, list, tuple)):
            raise TypeError(
                "results must be a dict (name -> image) or list/tuple of images"
            )

        if isinstance(results, dict):
            names = list(results.keys())
            images = list(results.values())
        else:
            names = [str(i) for i in range(len(results))]
            images = list(results)

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
