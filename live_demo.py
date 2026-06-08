import cv2
from image_loader import ImageLoader
from presenter import show_visualization, plot_confusion_matrix
from model import Model
from concurrent.futures import ProcessPoolExecutor


def nothing(x):
    pass


loader = ImageLoader()
images, masks, file_names, labels = loader.load_all()

model = Model((images, masks, labels), test_percentage=0.0)

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar("certeintyH", "Controls", 50, 100, nothing)
cv2.createTrackbar("certeintyS", "Controls", 50, 100, nothing)
cv2.createTrackbar("certeintyV", "Controls", 50, 100, nothing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")
while True:
    ret, frame = cap.read()

    if not ret:
        break

    certeintyH = cv2.getTrackbarPos("certeintyH", "Controls") / 100.0
    certeintyS = cv2.getTrackbarPos("certeintyS", "Controls") / 100.0
    certeintyV = cv2.getTrackbarPos("certeintyV", "Controls") / 100.0

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vis, features, prediction = model.Classify(
        image, [certeintyH, certeintyS, certeintyV]
    )
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
        cv2.imshow("Defects", vis["defects"])
    cv2.imshow("Live Classification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
