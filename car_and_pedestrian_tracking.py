from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("crossing road.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in ["car", "bus", "truck", "person"]:
                # Set color based on object type
                if label == "person":
                    color = (0, 0, 255)   # Red
                elif label == "car":
                    color = (0, 255, 0)   # Green
                elif label == "bus":
                    color = (255, 0, 0)   # Blue
                elif label == "truck":
                    color = (0, 255, 255) # Yellow

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Vehicle & Person Detector", frame)

    key = cv2.waitKey(1)
    if key == 113 or key == 81:  # ASCII for 'q' or 'Q'
        break

video.release()
cv2.destroyAllWindows()
