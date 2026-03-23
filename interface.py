import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from database import SessionLocal
from models import Teacher, Attendance
from datetime import datetime, timedelta
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Face Attendance System", layout="wide")

# Initialize InsightFace (Cached for performance)
@st.cache_resource
def load_face_models():
    app = FaceAnalysis(name="buffalo_l", root=".")
    app.prepare(ctx_id=-1)  # CPU Mode
    return app

face_app = load_face_models()

# State Management
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'cooldown' not in st.session_state:
    st.session_state.cooldown = {}

def change_page(page_name):
    st.session_state.page = page_name

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# SCREEN 1: DASHBOARD
# -----------------------------
if st.session_state.page == 'dashboard':
    st.title("🚀 Attendance Dashboard")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("➕ Register New Candidate", width="stretch"):
            change_page('register')
    with col_nav2:
        if st.button("📸 Mark Attendance", width="stretch"):
            change_page('attendance')

    st.divider()
    
    # Historical Logs Filtering
    st.subheader("Attendance History")
    selected_date = st.date_input("Select Date to View Logs", datetime.now())
    search_str = selected_date.strftime("%Y-%m-%d")
    
    db = SessionLocal()
    records = db.query(Attendance).filter(Attendance.date == search_str).all()
    
    if records:
        data = [{
            "Name": r.teacher.name,
            "Department": r.teacher.department,
            "In Time": r.in_time,
            "Out Time": r.out_time,
            "Status": r.status
        } for r in records]
        
        df = pd.DataFrame(data)
        st.dataframe(df, width="stretch")
        
        # CSV Export Feature
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to Excel (CSV)",
            data=csv,
            file_name=f"Attendance_{search_str}.csv",
            mime="text/csv",
        )
    else:
        st.info(f"No records found for {search_str}.")
    db.close()

# -----------------------------
# SCREEN 2: REGISTRATION
# -----------------------------
elif st.session_state.page == 'register':
    st.title("📝 Candidate Registration")
    if st.button("⬅ Back to Dashboard"): change_page('dashboard')
    
    col_reg1, col_reg2 = st.columns([1, 1])
    with col_reg1:
        name = st.text_input("Full Name")
        dept = st.text_input("Department")
        
    with col_reg2:
        img_file = st.camera_input("Capture Face")
    
    if img_file and name and dept:
        if st.button("Save Candidate Profile"):
            bytes_data = img_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            
            faces = face_app.get(rgb_img)
            if faces:
                emb_bytes = faces[0].embedding.astype(np.float32).tobytes()
                db = SessionLocal()
                new_teacher = Teacher(name=name, department=dept, embedding=emb_bytes)
                db.add(new_teacher)
                db.commit()
                db.close()
                st.success(f"✅ {name} registered successfully!")
            else:
                st.error("❌ Face detection failed. Ensure good lighting.")

# -----------------------------
# SCREEN 3: LIVE ATTENDANCE
# -----------------------------
elif st.session_state.page == 'attendance':
    st.title("🎥 Live Attendance Scanning")
    if st.button("⬅ Stop & Return to Dashboard"): change_page('dashboard')
    
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    db = SessionLocal()
    teachers = db.query(Teacher).all()

    while st.session_state.page == 'attendance':
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_frame)
        
        for face in faces:
            embedding = face.embedding.astype(np.float32)
            best_match, best_score = None, 0

            for teacher in teachers:
                stored_emb = np.frombuffer(teacher.embedding, dtype=np.float32)
                score = cosine_similarity(embedding, stored_emb)
                if score > best_score:
                    best_score, best_match = score, teacher

            if best_score > 0.6 and best_match:
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")
                
                # 30-second logic to prevent spam
                last_marked = st.session_state.cooldown.get(best_match.id)
                if last_marked and (now - last_marked) < timedelta(seconds=30):
                    continue

                record = db.query(Attendance).filter(
                    Attendance.teacher_id == best_match.id, 
                    Attendance.date == today_str
                ).first()

                if not record:
                    new_entry = Attendance(
                        teacher_id=best_match.id, 
                        date=today_str, 
                        in_time=now.strftime("%H:%M:%S"), 
                        status="Present"
                    )
                    db.add(new_entry)
                    st.toast(f"✅ {best_match.name} IN")
                else:
                    record.out_time = now.strftime("%H:%M:%S")
                    st.toast(f"✅ {best_match.name} OUT")
                
                db.commit()
                st.session_state.cooldown[best_match.id] = now

                # UI Overlay
                box = face.bbox.astype(int)
                cv2.rectangle(rgb_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(rgb_frame, f"{best_match.name}", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(rgb_frame)
    
    cap.release()
    db.close()
    #python -m streamlit run interface.py