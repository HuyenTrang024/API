# API nhận diện biển số xe

API đọc biển số xe máy và ô tô từ ảnh. Hệ thống chạy hai bước:

1. **Phát hiện biển số** bằng YOLO, đồng thời phân loại xe máy hay ô tô.
2. **Đọc ký tự** trên biển số bằng [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) (model CCT, đạt độ chính xác 0.95 lúc huấn luyện).

Kèm theo là một **web app** hiển thị camera RTSP trực tiếp, có nút bấm để nhận diện biển số ngay trên khung hình đang xem.

## Cấu trúc thư mục

```
API/
├── api/                          # Server API (FastAPI, cổng 8000)
│   ├── app_yolov5.py             # Bản YOLOv5 (torch.hub), có ghi log thời gian xử lý. Docker chạy bản này
│   └── app_yolov8.py             # Bản ultralytics (YOLOv8 / YOLOv5u)
│
├── webapp/                       # Giao diện xem camera (Flask, cổng 5000)
│   ├── app.py                    # Đọc RTSP bằng ffmpeg, stream MJPEG, gọi API nhận diện
│   └── templates/index.html
│
├── models/
│   ├── detection/                # Model YOLO phát hiện biển số
│   │   ├── best.pt               # ultralytics 8.3 – mặc định của app_yolov8
│   │   ├── best_5n_Q.pt          # YOLOv5n định dạng gốc – mặc định của app_yolov5
│   │   ├── best_5n_1.pt          # YOLOv5nu (ultralytics)
│   │   ├── best_5n_T.pt          # YOLOv5nu (ultralytics)
│   │   └── best_8n.pt            # YOLOv8n (ultralytics)
│   └── ocr/
│       └── ckpt-epoch_12-acc_0.950.onnx   # Model OCR đọc ký tự
│
├── configs/
│   ├── ocr_plate_config.yaml     # Bộ ký tự, số ký tự tối đa (9), kích thước ảnh 128×64
│   ├── ocr_model_config.yaml     # Kiến trúc model CCT (dùng khi huấn luyện)
│   └── ocr_augmentation.yaml     # Tăng cường dữ liệu khi huấn luyện OCR
│
├── training/
│   └── train_colab.py            # Notebook Colab: tăng cường dữ liệu, train YOLO, train OCR, xuất ONNX
│
├── Dockerfile
└── requirements.txt
```

## Cài đặt

**Yêu cầu:** Python 3.10, nên có GPU NVIDIA với CUDA 11.8 (không có GPU thì chạy bằng CPU, chậm hơn).

```bash
git clone https://github.com/HuyenTrang024/API.git
cd API
pip install -r requirements.txt
```

## Chạy API

```bash
# Bản YOLOv5
uvicorn api.app_yolov5:app --host 0.0.0.0 --port 8000

# Bản YOLOv8
uvicorn api.app_yolov8:app --host 0.0.0.0 --port 8000
```

Muốn dùng model khác thì đặt biến môi trường `DETECT_MODEL`. Model phải đúng định dạng với bản đang chạy:

```bash
DETECT_MODEL=models/detection/best_8n.pt uvicorn api.app_yolov8:app --port 8000
```

Chạy bằng Docker:

```bash
docker build -t plate-api .
docker run --gpus all -p 8000:8000 plate-api
```

## Gọi API

`POST /predict`, gửi ảnh dạng `multipart/form-data` trong trường `image`:

```bash
curl -X POST -F "image=@xe.jpg" http://localhost:8000/predict
```

Kết quả trả về:

```json
{
  "results": [
    { "plate": "29A12345", "box": [412, 380, 560, 430], "vehicle_type": "car" }
  ]
}
```

| Trường | Ý nghĩa |
|---|---|
| `plate` | Ký tự đọc được trên biển số |
| `box` | Toạ độ biển số trong ảnh `[x1, y1, x2, y2]` (pixel) |
| `vehicle_type` | `car` hoặc `motorbike` |

Nếu ảnh không có biển số, API trả mã `404` kèm `{"message": "Không phát hiện đối tượng nào."}`.

Có thể thử API trực tiếp trên trình duyệt tại `http://localhost:8000/docs`.

Bản YOLOv5 còn ghi thời gian xử lý từng bước (giải mã ảnh, YOLO, OCR) vào `logs/prediction.jsonl`.

## Chạy web app xem camera

Cần cài `ffmpeg`. Mở `webapp/app.py`, sửa `LPR_API_URL` (địa chỉ API) và danh sách `CAMERAS` (link RTSP), rồi chạy:

```bash
pip install flask requests
python webapp/app.py
```

Sau đó mở `http://localhost:5000`.

## Huấn luyện lại

`training/train_colab.py` được xuất từ Google Colab và đọc dữ liệu trên Google Drive (`/content/drive/MyDrive/CPC1HN/...`). Nên mở và chạy từng bước trên Colab, gồm:

1. Tăng cường dữ liệu ảnh xe (lật, xoay, nghiêng, độ sáng, làm mờ, nhiễu).
2. Train YOLO phát hiện biển số.
3. Cắt ảnh biển số, rồi train OCR với `configs/ocr_model_config.yaml` và `configs/ocr_augmentation.yaml`.
4. Xuất model OCR sang ONNX và chép vào `models/ocr/`.
