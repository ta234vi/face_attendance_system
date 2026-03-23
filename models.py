from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, LargeBinary
from sqlalchemy.orm import relationship
from database import Base


class Teacher(Base):
    __tablename__ = "teachers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String)
    embedding = Column(LargeBinary, nullable=False)

    attendance_records = relationship("Attendance", back_populates="teacher")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=False)

    date = Column(String, nullable=False)
    in_time = Column(String, nullable=True)   # nullable now
    out_time = Column(String, nullable=True)
    status = Column(String, default="Present")

    teacher = relationship("Teacher", back_populates="attendance_records")

    __table_args__ = (
        UniqueConstraint("teacher_id", "date", name="unique_teacher_date"),
    )