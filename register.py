import cv2
import numpy as np
from insightface.app import FaceAnalysis
from models import Teacher
from database import SessionLocal, Base, engine

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # -1 = CPU (safer)

name = input("Enter Teacher Name: ")
department = input("Enter Department: ")

cap = cv2.VideoCapture(0)

print("Press S to capture face")
print("Press Q to quit")

embedding_bytes = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) > 0:
            # Convert embedding to float32 and then to bytes
            embedding_bytes = faces[0].embedding.astype(np.float32).tobytes()
            print("✅ Face captured!")
            break
        else:
            print("❌ No face detected!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if embedding_bytes:
    db = SessionLocal()

    teacher = Teacher(
        name=name,
        department=department,
        embedding=embedding_bytes   # ← SAVE BYTES
    )

    db.add(teacher)
    db.commit()
    db.close()

    print("✅ Teacher registered successfully!")
else:
    print("❌ Registration failed.")