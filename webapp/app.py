## import thư viện 
import os
import requests
import urllib3
import cv2
import threading
import time
import subprocess
import numpy as np
from flask import Flask, jsonify, render_template, Response
from collections import deque

### Tắt cảnh báo SSL không an toàn (do requests tới API HTTP/HTTPS)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- API và Camera ---
LPR_API_URL = "http://alpr.icpc1hn.work/predict"   ### URL API nhận diện biển số

### Danh sách camera RTSP
CAMERAS = [
    {
        "id": 1,
        "name": "Xe máy vào 1",
        "rtsp": "rtsp://admin:Cpc1hn2025@@192.168.6.200:554/Streaming/Channels/101",  # link RTSP
        "width": 1280,    # độ rộng ảnh
        "height": 720,    # độ cao ảnh
    },
]

### Khởi tạo Flask app
app = Flask(__name__)

### Tạo queue chứa frame cho từng camera (lưu tối đa 5 frame)
frame_queues = {cam["id"]: deque(maxlen=5) for cam in CAMERAS}


# ---- Hàm đọc camera bằng ffmpeg pipe ----
def capture_frames_ffmpeg(camera_id):
    cam = next(c for c in CAMERAS if c["id"] == camera_id)  ### lấy thông tin camera theo id
    width, height = cam["width"], cam["height"]

    ### Lệnh ffmpeg: đọc RTSP -> xuất raw frame (BGR)
    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",    # dùng TCP để ổn định hơn
        "-i", cam["rtsp"],           # input RTSP URL
        "-f", "rawvideo",            # xuất video dạng raw
        "-pix_fmt", "bgr24",         # định dạng pixel (OpenCV dùng BGR)
        "-vf", f"scale={width}:{height}",  # resize khung hình
        "-"
    ]

    while True:
        try:
            print(f" Bắt đầu đọc camera {camera_id} ...")
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,   # tránh log quá nhiều
                bufsize=10**8
            )
            frame_count = 0
            while True:
                ### đọc đúng số byte cho 1 frame (width*height*3 kênh màu)
                raw_frame = proc.stdout.read(width * height * 3)
                if not raw_frame:
                    print(f" Camera {camera_id} không còn dữ liệu, restart ffmpeg...")
                    break
                ### chuyển buffer thành numpy array (ảnh OpenCV)
                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                frame_queues[camera_id].append(frame)  # đưa frame vào queue
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f" Camera {camera_id} đã nhận {frame_count} frames")
        except Exception as e:
            print(f" Camera {camera_id} error: {e}")
            time.sleep(2)  # đợi rồi reconnect nếu lỗi


# ---- Stream video MJPEG ----
def generate_frames(camera_id):
    while True:
        if len(frame_queues[camera_id]) > 0:
            frame = frame_queues[camera_id][-1]  # lấy frame mới nhất
            ret, buffer = cv2.imencode(".jpg", frame)  # mã hóa thành JPEG
            frame_bytes = buffer.tobytes()
            ### Trả về theo chuẩn MJPEG (multipart/x-mixed-replace)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        else:
            time.sleep(0.1)

# ---- API nhận diện sync ----
@app.route("/predict_plate/<int:camera_id>", methods=["POST"])
def predict_plate(camera_id):
    frame = None
    if len(frame_queues[camera_id]) > 0:
        frame = frame_queues[camera_id][-1].copy()  ### copy frame hiện tại

    if frame is None:
        print(" Lỗi: Không có frame nào từ camera.")
        return jsonify({"error": "Không có frame nào từ camera."}), 500

    ### Mã hóa frame thành JPEG bytes
    ret, buffer = cv2.imencode(".jpg", frame)
    image_bytes = buffer.tobytes()

    try:
        files = {"image": ("snapshot.jpg", image_bytes, "image/jpeg")}
        print(f" Bước 1: Gửi ảnh từ camera {camera_id} tới API...")
        resp = requests.post(LPR_API_URL, files=files, timeout=30)
        
        print(f" Bước 2: Đã nhận được phản hồi từ API. Mã trạng thái: {resp.status_code}")
        resp.raise_for_status()
        
        print(" Bước 3: Phản hồi có mã trạng thái OK. Bắt đầu xử lý JSON.")
        
        ### Debug: in raw response text để kiểm tra
        print(" Nội dung phản hồi (body):", resp.text)
        
        result = resp.json()  ### parse JSON
        print(" Bước 4: Xử lý JSON thành công. Kết quả:", result)

        ### Nếu có biển số trong response
        if "plates" in result and len(result["plates"]) > 0:
            plate_text = result["plates"][0]["plate_text"]
            vehicle_type = result["plates"][0].get("vehicle_type", "N/A")
            print(f" Kết quả nhận diện thành công: {plate_text}, Loại xe: {vehicle_type}")
            return jsonify({"plate_text": plate_text, "vehicle_type": vehicle_type})
        else:
            print(" Không tìm thấy biển số trong phản hồi.")
            return jsonify({"error": "Không tìm thấy biển số."})
    except requests.exceptions.HTTPError as errh:
        print(f" Lỗi HTTP xảy ra: {errh}")
        return jsonify({"error": f"Lỗi HTTP: {str(errh)}"}), 500
    except requests.exceptions.ConnectionError as errc:
        print(f" Lỗi kết nối xảy ra: {errc}")
        return jsonify({"error": f"Lỗi kết nối: {str(errc)}"}), 500
    except requests.exceptions.Timeout as errt:
        print(f" Lỗi timeout xảy ra: {errt}")
        return jsonify({"error": f"Lỗi timeout: {str(errt)}"}), 500
    except requests.exceptions.RequestException as err:
        print(f" Lỗi Request không xác định: {err}")
        return jsonify({"error": f"Lỗi Request không xác định: {str(err)}"}), 500
    except Exception as e:
        ### Bắt tất cả lỗi khác (VD: JSONDecodeError)
        print(f" Lỗi không xác định khi xử lý phản hồi: {e}")
        return jsonify({"error": f"Lỗi không xác định: {str(e)}"}), 500


# ---- Flask Routes ----
@app.route("/")
def home():
    ### render template index.html, truyền danh sách camera
    return render_template("index.html", cameras=CAMERAS)


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    ### endpoint stream MJPEG
    return Response(generate_frames(camera_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- Main ----
if __name__ == "__main__":
    ### Tạo thread cho mỗi camera -> đọc frame song song
    for cam in CAMERAS:
        t = threading.Thread(target=capture_frames_ffmpeg, args=(cam["id"],))
        t.daemon = True
        t.start()

    ### Chạy Flask server
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
