# 👁️ DivyaDrishti — AI Currency & Object Detector

A real-time assistive tool designed for the visually impaired to identify Indian currency (coins and notes) and common household objects using AI.

---

## 🚀 Key Features

- **Unified Currency Detection**: Specialized YOLO-trained model for all Indian currency:
  - **Notes**: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500
  - **Coins**: ₹1, ₹2, ₹5, ₹10, ₹20
- **Object Detection**: Identifies 80+ common objects (from COCO dataset) using YOLOv8.
- **Natural Voice Feedback**: Built-in Text-to-Speech (TTS) that announces detected items.
- **Smart Persistence Logic**: Uses a 3-second delay to ensure objects are stable before announcing, preventing false triggers.
- **Web Interface**: Real-time streaming server with a clean, responsive UI.

---

## 🛠️ Project Structure

- `server.py`: The heart of the project. A Flask server that handles video streaming, AI processing (Object + Currency), and TTS announcements.
- `train_currency.py`: The training pipeline used to build the unified currency classifier.
- `currency_best.pt`: The production-ready trained YOLOv8 model for currency.
- `yolov8n.pt`: The base YOLOv8 model for general object detection.
- `static/` & `templates/`: Modern web interface assets.

---

## 💻 Setup & Installation

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU (Optional, but recommended for high FPS)

### 2. Install Dependencies
```bash
pip install ultralytics flask opencv-python pyttsx3 numpy
```

### 3. Run the Server
```bash
python server.py
```
After running, open your browser and go to: `http://localhost:5001` or the network IP address shown in the terminal.

---

## 📖 Usage

1. **Start the Stream**: Open the web app on your phone or laptop.
2. **Detection**: Point the camera at a note, coin, or object.
3. **Wait for Confirmation**: Hold the object steady for **3 seconds**. The system will highlight confirmed objects with a green box.
4. **Voice Output**: The system will automatically announce the name of the confirmed object/currency.

---

## 🤖 Model Performance

The custom currency classifier was trained on a balanced dataset of Indian currency and achieved:
- **Accuracy**: 100% on the validation set.
- **Top-1 Confidence**: Average >95% for all denominations.

---

## 👥 Contributors
- Developed as part of a group project for visually impaired accessibility.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
