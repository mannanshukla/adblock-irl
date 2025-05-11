import cv2
from ultralytics import YOLO

# —— 1) Load your local billboard detector checkpoint ——
model = YOLO("yolo11l.pt")

# —— 2) Open your camera (swap 0→1→2… until you hit the right device) ——
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Try changing the index in VideoCapture().")

# Size of each “pixel block” for the mosaic censor
pixel_size = 15

# —— 3) Live inference + mosaic censor loop ——
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run inference; stream=True keeps things hot, conf=0.25 filters low-score boxes
    for result in model(frame, stream=True, conf=0.25):
        # Extract Nx4 array of xyxy boxes
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int)

        # For each detected billboard, apply the blocky mosaic
        for (x1, y1, x2, y2) in bboxes:
            # Clamp to valid ROI
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Downsample to small “pixel grid”
            small = cv2.resize(
                roi,
                (max(1, (x2 - x1) // pixel_size), max(1, (y2 - y1) // pixel_size)),
                interpolation=cv2.INTER_LINEAR
            )
            # Upsample back to original ROI size using nearest-neighbor
            mosaic = cv2.resize(
                small,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )
            # Replace the region in the frame
            frame[y1:y2, x1:x2] = mosaic

    # Display the censored feed
    cv2.imshow("Live Billboard Mosaic Censor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
