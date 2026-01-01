from ultralytics import YOLO
import cv2
import easyocr
import time

# Load your trained model
model = YOLO("ANPR_Training/anpr_yolo11n/weights/best.pt")

# Initialize EasyOCR reader (English + digits, no GPU needed)
reader = easyocr.Reader(['en'], gpu=False)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Try a different camera index (e.g., 1 or 2).")
    exit()

# Set resolution for better performance (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 Real-time ANPR started! Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run YOLO detection
    results = model(frame, imgsz=640, verbose=False)[0]

    plate_texts = []

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        plate_crop = frame[y1:y2, x1:x2]

        # OCR on cropped plate
        ocr_result = reader.readtext(plate_crop, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ')
        text = ' '.join(ocr_result).strip() if ocr_result else "???"

        plate_texts.append(text)

        # Draw box and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show detected plates in title
    title = f"Plates: {', '.join(plate_texts) if plate_texts else 'None'}"
    cv2.imshow("Real-Time ANPR - Press 'q' to quit", frame)
    cv2.setWindowTitle("Real-Time ANPR - Press 'q' to quit", title)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam ANPR stopped. Goodbye! 👋")