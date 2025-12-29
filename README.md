

# inference-yolo-onnx-models 🎯

Ứng dụng này cung cấp **giao diện trực quan** để thực hiện **YOLO inference** với các mô hình đã được convert sang định dạng **ONNX**.  
Thay vì phải chạy lệnh thủ công, bạn có thể sử dụng giao diện để tải mô hình, chọn ảnh/video và xem kết quả nhận diện ngay lập tức.

---

## 📂 Cấu trúc repo

```
inference-yolo-onnx-models/
│── .idea/                  # Cấu hình IDE
│── runs/detect/            # Kết quả inference lưu lại
│── best.onnx               # Mô hình YOLO ONNX
│── best.pt                 # Mô hình PyTorch gốc
│── best_float32.tflite     # Mô hình TFLite (float32)
│── best_int8.tflite        # Mô hình TFLite (int8)
│── coco8.yaml              # Dataset cấu hình
│── detect.py               # Script inference YOLO
│── main.py                 # Entry point ứng dụng giao diện
│── requirements.txt        # Thư viện cần thiết
│── result.jpg              # Ví dụ kết quả inference
│── test.py                 # Script test nhanh
│── yolo11n-pose.onnx       # Mô hình YOLO pose ONNX
│── yolotflite.py           # Script inference TFLite
```

---

## ⚙️ Cài đặt

1. Clone repo:
   ```bash
   git clone https://github.com/Techsolutions2024/inference-yolo-onnx-models.git
   cd inference-yolo-onnx-models
   ```

2. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Chạy ứng dụng giao diện

Chạy file `main.py` để mở giao diện:

```bash
python main.py
```

Trong giao diện, bạn có thể:
- Chọn mô hình ONNX (`best.onnx`, `yolo11n-pose.onnx`, …).  
- Tải ảnh hoặc video đầu vào.  
- Xem kết quả nhận diện trực tiếp trên màn hình.  
- Lưu kết quả inference vào thư mục `runs/detect/`.

---

## 📌 Các tính năng chính

- Hỗ trợ **YOLO ONNX inference** qua [onnxruntime](https://onnxruntime.ai/).  
- Giao diện trực quan, dễ sử dụng.  
- Nhận diện vật thể từ ảnh hoặc video.  
- Hỗ trợ **pose estimation** với YOLO pose.  
- Xuất kết quả inference ra file ảnh/video.  

---

## 🧩 Ví dụ sử dụng script

Nếu muốn chạy trực tiếp bằng script:

```bash
python detect.py --source test.jpg --model best.onnx
```

---

## 📖 Hướng phát triển

- Thêm hỗ trợ nhiều phiên bản YOLO (YOLOv5, YOLOv8, YOLOv11).  
- Tích hợp lựa chọn CPU/GPU trong giao diện.  
- Thêm tính năng benchmark tốc độ inference.  

---

## 📜 License

MIT License – bạn có thể sử dụng, chỉnh sửa và phát triển repo này cho mục đích cá nhân hoặc thương mại.
