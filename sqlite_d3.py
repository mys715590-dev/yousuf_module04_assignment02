"""Create the SQLite academic database from synthetic_students_150.csv."""
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "synthetic_students_150.csv"
DB_PATH = BASE_DIR / "student.db"

REQUIRED_COLUMNS = [
    "student_id", "name", "class", "section", "marks",
    "attendance_pct", "city", "enrollment_month",
]


def create_database() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH.name}")

    data = pd.read_csv(CSV_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()
    data["student_id"] = pd.to_numeric(data["student_id"], errors="raise").astype(int)
    data["marks"] = pd.to_numeric(data["marks"], errors="raise").astype(int)
    data["attendance_pct"] = pd.to_numeric(data["attendance_pct"], errors="raise").astype(int)

    if data["student_id"].duplicated().any():
        raise ValueError("student_id values must be unique.")
    if not data["marks"].between(0, 100).all():
        raise ValueError("Marks must be between 0 and 100.")
    if not data["attendance_pct"].between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100.")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DROP TABLE IF EXISTS student")
        connection.execute("""
            CREATE TABLE student (
                student_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                class TEXT NOT NULL,
                section TEXT NOT NULL,
                marks INTEGER NOT NULL CHECK (marks BETWEEN 0 AND 100),
                attendance_pct INTEGER NOT NULL CHECK (attendance_pct BETWEEN 0 AND 100),
                city TEXT NOT NULL,
                enrollment_month TEXT NOT NULL
            )
        """)
        data.to_sql("student", connection, if_exists="append", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_student_class ON student(class)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_student_section ON student(section)")
        connection.commit()

    return len(data)


if __name__ == "__main__":
    count = create_database()
    print(f"Database created successfully: {DB_PATH.name} ({count} students)")