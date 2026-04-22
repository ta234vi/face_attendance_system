import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from database import SessionLocal, engine  # Added engine
from models import Teacher, Attendance, Base  # Added Base
from datetime import datetime, timedelta
import pandas as pd

# --- CRITICAL: FIX FOR OPERATIONAL ERROR ---
# This creates the tables in the database if they don't exist on the server
Base.metadata.create_all(bind=engine)

# Page Configuration
st.set_page_config(page_title="Face Attendance System", layout="wide")

# Initialize InsightFace (Cached for performance)
@st.cache_resource
def load_face_models():
    # root="." ensures it looks for models in the current directory
    app = FaceAnalysis(name="buffalo_l", root=".")
    app.prepare(ctx_id=-1)  # CPU Mode for Streamlit Cloud
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
        if st.button("➕ Register New Candidate", use_container_width=True):
            change_page('register')
    with col_nav2:
        if st.button("📸 Mark Attendance", use_container_width=True):
            change_page('attendance')

    st.divider()
    
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
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to CSV",
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
        img_file = st.camera_input("Capture Face for Registration")
    
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
                st.error("❌ Face detection failed. Ensure your face is clear.")

# -----------------------------
# SCREEN 3: LIVE ATTENDANCE (Cloud Version)
# -----------------------------
elif st.session_state.page == 'attendance':
    st.title("📸 Mark Your Attendance")
    if st.button("⬅ Back to Dashboard"): change_page('dashboard')
    
    # Use camera_input instead of cv2.VideoCapture for Cloud Deployment
    img_file = st.camera_input("Scan your face")
    
    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        faces = face_app.get(rgb_frame)
        
        if not faces:
            st.warning("No face detected. Please try again.")
        else:
            db = SessionLocal()
            teachers = db.query(Teacher).all()
            
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
                    
                    # 30-second cooldown check
                    last_marked = st.session_state.cooldown.get(best_match.id)
                    if last_marked and (now - last_marked) < timedelta(seconds=30):
                        st.info(f"Attendance already marked for {best_match.name} recently.")
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
                        st.success(f"✅ Welcome {best_match.name}! (IN Marked)")
                    else:
                        record.out_time = now.strftime("%H:%M:%S")
                        st.success(f"✅ Goodbye {best_match.name}! (OUT Marked)")
                    
                    db.commit()
                    st.session_state.cooldown[best_match.id] = now
                else:
                    st.error("Unknown person detected. Please register first.")
            db.close()
