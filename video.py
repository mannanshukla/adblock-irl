import cv2
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(
        description="Censor detected billboards in a video with a mosaic effect."
    )
    p.add_argument("input",  help="Path to input video file")
    p.add_argument("output", help="Path to write censored output video")
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detection"
    )
    p.add_argument(
        "--pixel_size", type=int, default=15,
        help="Block size for the mosaic censor"
    )
    args = p.parse_args()

    # Load your local billboard detector
    model = YOLO("yolo11l.pt")

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    # Prepare output writer (match input FPS and frame size)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        for result in model(frame, stream=True, conf=args.conf):
            # result.boxes.xyxy is an N×4 array of [x1,y1,x2,y2]
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int)

            # Apply mosaic censor to each detected billboard region
            for (x1, y1, x2, y2) in bboxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Downsample & upsample for the blocky effect
                small = cv2.resize(
                    roi,
                    ((x2 - x1) // args.pixel_size or 1,
                     (y2 - y1) // args.pixel_size or 1),
                    interpolation=cv2.INTER_LINEAR
                )
                mosaic = cv2.resize(
                    small, (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST
                )
                frame[y1:y2, x1:x2] = mosaic

        # Write and show
        out.write(frame)
        cv2.imshow("Censored Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
