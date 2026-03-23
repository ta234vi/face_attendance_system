from flask import Flask, render_template
from database import SessionLocal, engine
from models import Base, Teacher, Attendance

Base.metadata.create_all(bind=engine)

app = Flask(__name__)
db = SessionLocal()

@app.route("/")
def dashboard():
    teachers = db.query(Teacher).all()
    attendance = db.query(Attendance).all()
    return render_template("dashboard.html", teachers=teachers, attendance=attendance)

if __name__ == "__main__":
    app.run(debug=True)