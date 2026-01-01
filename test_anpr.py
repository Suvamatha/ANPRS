from ultralytics import YOLO
import cv2
import glob
import easyocr
import os

# Load model and OCR reader
model = YOLO("ANPR_Training/anpr_yolo11n/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)  # 'en' for English plates

# Create results folder
os.makedirs("results", exist_ok=True)

test_images = glob.glob("test/images/*.jpg")
print(f"Found {len(test_images)} test images. Starting ANPR + OCR...")

for img_path in test_images:  # Change to test_images for all
    results = model(img_path, imgsz=640)[0]
    img = cv2.imread(img_path)
    annotated = img.copy()

    plate_texts = []

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        plate_crop = img[y1:y2, x1:x2]

        # Read text from cropped plate
        ocr_result = reader.readtext(plate_crop, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ')
        text = ' '.join(ocr_result).strip() if ocr_result else "???"

        plate_texts.append(text)

        # Draw box and text on image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    print(f"{os.path.basename(img_path)} → Detected text: {plate_texts}")

    # Show for 3 seconds
    cv2.namedWindow("ANPR + OCR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANPR + OCR", 1000, 700)
    cv2.imshow("ANPR + OCR", annotated)
    key = cv2.waitKey(3000) & 0xFF
    if key == ord('q'):
        break

    # Save result
    filename = os.path.basename(img_path)
    cv2.imwrite(f"results/{filename}", annotated)

cv2.destroyAllWindows()
print("Done! Check 'results' folder — now with plate text!")