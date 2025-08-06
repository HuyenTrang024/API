from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("best.pt")
yolo_model.to(device)  


plate_recognizer = LicensePlateRecognizer(
    onnx_model_path="ckpt-epoch_12-acc_0.950.onnx",
    plate_config_path="cct_xs_v1_global_plate_config.yaml",
)

# --- Hàm đọc ảnh ---
def read_imagefile(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = read_imagefile(image.file)

    # --- Dự đoán bằng YOLOv8 ---
    results = yolo_model.predict(
        source=img,
        conf=0.25,
        save=False,
        verbose=False,
        device=0 if torch.cuda.is_available() else 'cpu'  
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return JSONResponse(status_code=404, content={"message": "Không phát hiện đối tượng nào."})

    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])

        # Crop biển số
        cropped_plate = img[y1:y2, x1:x2]

        # OCR
        ocr_result = plate_recognizer.run(cropped_plate)
        plate_text = ocr_result

        vehicle_type = "motorbike" if cls_id == 1 else "car"

        output.append({
            "plate": plate_text,
            "box": [x1, y1, x2, y2],
            "vehicle_type": vehicle_type,
        })

    return {"results": output}

if __name__ == "__main__":
    uvicorn.run("connect_API:app", host="0.0.0.0", port=8000, reload=False)
