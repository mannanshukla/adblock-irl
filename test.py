import cv2
from ultralytics import YOLO

# Load the tiny pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Open your Mac’s default webcam (device index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open webcam')

# Size of each pixel block in the mosaic
pixel_size = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference in streaming mode (one frame at a time)
    for result in model(frame, stream=True, conf=0.25):
        boxes   = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores  = result.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, scores):
            # If this is a person (class 0), pixelate that region
            if cls == 0:
                roi = frame[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                # downscale
                small = cv2.resize(roi, (max(1, w//pixel_size), max(1, h//pixel_size)), interpolation=cv2.INTER_LINEAR)
                # upscale back to original size using nearest neighbor
                mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = mosaic

            # Draw the bounding box and confidence label
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the pixelated frame
    cv2.imshow('YOLOv8 Webcam – Cartoon Censor', frame)
    if cv2.waitKey(1) == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
