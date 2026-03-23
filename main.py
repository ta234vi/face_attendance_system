import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from database import engine, SessionLocal, Base
from models import Teacher, Attendance

Base.metadata.create_all(bind=engine)

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # -1 = CPU

db = SessionLocal()

video = cv2.VideoCapture(0)

print("Live Teacher Attendance Started...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    faces = app.get(frame)

    # Reload teachers each loop (important if new teachers added)
    teachers = db.query(Teacher).all()

    for face in faces:
        input_embedding = face.embedding

        for teacher in teachers:
            # Convert stored bytes to numpy array
            stored_embedding = np.frombuffer(teacher.embedding, dtype=np.float32)

            # Cosine similarity
            similarity = np.dot(stored_embedding, input_embedding) / (
                np.linalg.norm(stored_embedding) * np.linalg.norm(input_embedding)
            )

            if similarity > 0.6:
                today = datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.now().strftime("%H:%M:%S")

                record = db.query(Attendance).filter(
                    Attendance.teacher_id == teacher.id,
                    Attendance.date == today
                ).first()

                if not record:
                    # First scan today → IN time
                    new_record = Attendance(
                        teacher_id=teacher.id,
                        date=today,
                        in_time=now_time,
                        out_time=None,
                        status="Present"
                    )
                    db.add(new_record)
                    db.commit()
                    print(f"{teacher.name} - IN Time Marked")

                else:
                    # Update OUT time (overwrite allowed)
                    record.out_time = now_time
                    db.commit()
                    print(f"{teacher.name} - OUT Time Updated")

    cv2.imshow("Teacher Attendance", frame)

    # Press ENTER to exit
    if cv2.waitKey(1) & 0xFF == 13:
        break

video.release()
cv2.destroyAllWindows()