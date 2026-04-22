
# Face Attendance System

A real-time biometric attendance solution built with Python and Streamlit. This system uses Deep Learning to identify faces and log attendance automatically.

## 🚀 Features
* **Real-time Detection:** High-speed face detection using OpenCV.
* **Biometric Accuracy:** Powered by InsightFace (512D embeddings) for precise identity verification.
* **Automated Logging:** Instant check-in/out records saved to a SQLite database.
* **Reporting:** Export attendance logs directly to CSV files.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Face Logic:** InsightFace, ONNX Runtime, OpenCV
* **Database:** SQLAlchemy (SQLite)
* **Language:** Python

## 📦 Installation & Deployment
This project is configured for **Streamlit Cloud**.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ta234vi/face_attendance_system.git](https://github.com/ta234vi/face_attendance_system.git)
2. pip install -r requirements.txt

3. streamlit run interface.py
