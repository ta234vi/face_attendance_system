import cv2
import numpy as np
from insightface.app import FaceAnalysis
from database import SessionLocal, engine, Base
from models import Teacher, Attendance
from datetime import datetime, timedelta
import sys

# Initialize DB
Base.metadata.create_all(bind=engine)

# Initialize Face Model - buffalo_l is high accuracy
# ctx_id=0 for GPU, -1 for CPU
face_app = FaceAnalysis(name="buffalo_l", root=".")
face_app.prepare(ctx_id=-1) 

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_attendance():
    db = SessionLocal()
    # Load teachers once to save CPU/Memory
    teachers_list = db.query(Teacher).all()
    cooldown = {}
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera error")
        return

    print("✅ System Active. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Convert BGR to RGB (InsightFace requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Detect & Recognize
        faces = face_app.get(rgb_frame)
        
        for face in faces:
            embedding = face.embedding.astype(np.float32)
            best_match = None
            best_score = 0

            for teacher in teachers_list:
                stored_emb = np.frombuffer(teacher.embedding, dtype=np.float32)
                score = cosine_similarity(embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_match = teacher

            # 3. Threshold check (0.6 is standard for buffalo_l)
            if best_score > 0.6:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                
                # Cooldown to prevent duplicate logs (30 seconds)
                if best_match.id in cooldown:
                    if now - cooldown[best_match.id] < timedelta(seconds=30):
                        continue

                # Database Logic
                existing = db.query(Attendance).filter(
                    Attendance.teacher_id == best_match.id, 
                    Attendance.date == today
                ).first()

                if not existing:
                    new_log = Attendance(teacher_id=best_match.id, date=today, in_time=now.strftime("%H:%M:%S"), status="Present")
                    db.add(new_log)
                    print(f"✔️ {best_match.name} marked IN")
                elif not existing.out_time:
                    existing.out_time = now.strftime("%H:%M:%S")
                    print(f"✔️ {best_match.name} marked OUT")
                
                db.commit()
                cooldown[best_match.id] = now

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    run_attendance()