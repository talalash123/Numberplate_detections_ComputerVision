import os
import cv2
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

def process_anpr_video(input_video_name='sample.mp4', weights_path='best.pt'):
    # 1. Model & EasyOCR Load
    if not os.path.exists(weights_path):
        print(f"❌ Error: '{weights_path}' file nahi mili! File ko project root folder mein rakhein.")
        return

    if not os.path.exists(input_video_name):
        print(f"❌ Error: Input video '{input_video_name}' nahi mili! Video file ko project root folder mein rakhein.")
        return

    print("⏳ Loading YOLOv8 Model & EasyOCR Reader...")
    model = YOLO(weights_path)
    reader = easyocr.Reader(['en'], gpu=False)  # Agar NVIDIA GPU ho toh gpu=True kar sakte hain

    # 2. Input Video Stream Setup
    cap = cv2.VideoCapture(input_video_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Output Video Writer Setup
    output_video_name = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    detections_log = []
    frame_count = 0

    print("🚀 Video Processing Started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO Inference (Confidence Threshold 0.4)
        results = model(frame, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Safety boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Plate Crop
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size > 0:
                    # EasyOCR Text Extraction
                    ocr_res = reader.readtext(plate_crop)
                    plate_text = "".join([text[1] for text in ocr_res if text[2] > 0.2]).strip()

                    # Bounding Box Draw Karein
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Text Overlay Draw Karein
                    if plate_text:
                        # Label background box
                        label_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        label_w, label_h = label_size
                        cv2.rectangle(frame, (x1, max(0, y1 - label_h - 10)), (x1 + label_w + 10, y1), (0, 255, 0), -1)
                        
                        # Text label
                        cv2.putText(frame, plate_text, (x1 + 5, max(label_h + 5, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                        # CSV Logging Record
                        detections_log.append({
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Frame_Number': frame_count,
                            'Plate_Number': plate_text
                        })

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    # Save Log to CSV
    if detections_log:
        df = pd.DataFrame(detections_log)
        # Unique plate numbers per timestamp / continuous log
        df.to_csv('detections.csv', index=False)
        print("📊 Detection log saved to 'detections.csv'")

    print(f"\n✅ Processing Complete! Output video saved as '{output_video_name}'.")

if __name__ == "__main__":
    # Aapki sample video file ka name
    process_anpr_video(input_video_name='sample.mp4', weights_path='best.pt')