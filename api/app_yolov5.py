from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
import time
import os
import json
from datetime import datetime
from fast_plate_ocr import LicensePlateRecognizer

app = FastAPI()

# Đường dẫn tính từ thư mục gốc repo, chạy ở thư mục nào cũng được
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECT_MODEL = os.environ.get("DETECT_MODEL", os.path.join(ROOT_DIR, "models/detection/best_5n_Q.pt"))
OCR_MODEL = os.path.join(ROOT_DIR, "models/ocr/ckpt-epoch_12-acc_0.950.onnx")
OCR_PLATE_CONFIG = os.path.join(ROOT_DIR, "configs/ocr_plate_config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model 
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECT_MODEL, force_reload=True)
yolo_model.to(device)
yolo_model.eval()

# Load OCR model
plate_recognizer = LicensePlateRecognizer(
    onnx_model_path=OCR_MODEL,
    plate_config_path=OCR_PLATE_CONFIG,
)

# Hàm đọc ảnh
def read_imagefile(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

# Hàm lưu JSONL
def save_to_jsonl(data, timing, filename="prediction.jsonl"):
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", filename)
    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plate": item["plate"],
                "box": item["box"],
                "vehicle_type": item["vehicle_type"],
                "timing": item["timing"],
                "overall": timing
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    total_start = time.time()

    # Đọc ảnh
    decode_start = time.time()
    img = read_imagefile(image.file)
    decode_duration = time.time() - decode_start

    # Dự đoán với YOLO 
    yolo_start = time.time()
    results = yolo_model(img)  
    yolo_duration = time.time() - yolo_start

    df = results.pandas().xyxy[0]
    if df.empty:
        return JSONResponse(status_code=404, content={"message": "Không phát hiện đối tượng nào."})

    output = []

    for idx, row in df.iterrows():
        crop_start = time.time()
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cls_id = int(row['class'])

        cropped_plate = img[y1:y2, x1:x2]
        crop_duration = time.time() - crop_start

        ocr_start = time.time()
        plate_text = plate_recognizer.run(cropped_plate)
        ocr_duration = time.time() - ocr_start

        vehicle_type = "car" if cls_id == 1 else "motorbike"

        output.append({
            "plate": plate_text,
            "box": [x1, y1, x2, y2],
            "vehicle_type": vehicle_type,
            "timing": {
                "crop": f"{crop_duration:.3f}",
                "ocr": f"{ocr_duration:.3f}"
            }
        })

    total_duration = time.time() - total_start

    # Ghi file log JSONL
    overall_timing = {
        "decode": f"{decode_duration:.3f}",
        "yolo": f"{yolo_duration:.3f}",
        "total": f"{total_duration:.3f}"
    }
    save_to_jsonl(output, overall_timing)

    # Trả kết quả cho client 
    return {
        "results": [
            {
                "plate": item["plate"],
                "box": item["box"],
                "vehicle_type": item["vehicle_type"]
            }
            for item in output
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
