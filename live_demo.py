import cv2
from image_loader import ImageLoader
from presenter import show_visualization, plot_confusion_matrix
from model import Model
from concurrent.futures import ProcessPoolExecutor

loader = ImageLoader()
images, masks, file_names, labels = loader.load_all()

model = Model((images, masks, labels), test_percentage=0.0)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vis, features, prediction = model.Classify(image)
    cv2.putText(
        frame,
        f"Class: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    if len(features) != 0:
        cv2.imshow("Contour", vis["contour"])
        cv2.imshow("Defects", vis["defects"])
        cv2.imshow("Mask", vis["created_mask"])
    cv2.imshow("Live Classification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
