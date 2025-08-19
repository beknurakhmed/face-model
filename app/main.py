import os
import time
import argparse
import datetime as dt

import cv2
from deepface import DeepFace
from sqlalchemy import create_engine, text

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Face analytics (RTSP/Webcam → MySQL)")
parser.add_argument("--source", choices=["rtsp", "webcam"], default=os.getenv("SOURCE", "rtsp"))
parser.add_argument("--rtsp-url", default=os.getenv("RTSP_URL", ""))
parser.add_argument("--device", type=int, default=int(os.getenv("WEBCAM_INDEX", "0")))
parser.add_argument("--interval", type=float, default=float(os.getenv("INTERVAL_SEC", "1.0")))
parser.add_argument("--display", action="store_true", help="Show window (host only; not in headless docker)")
args = parser.parse_args()

# ---------- DB ----------
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB   = os.getenv("MYSQL_DB", "face_analytics")
MYSQL_USER = os.getenv("MYSQL_USER", "fa_user")
MYSQL_PWD  = os.getenv("MYSQL_PASSWORD", "fa_pass")

# engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4")
engine = create_engine("mysql+pymysql://fa_user:fa_pass@127.0.0.1:3306/fa_db")

with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS faces (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        ts DATETIME NOT NULL,
        source VARCHAR(32) NOT NULL,
        face_index INT NOT NULL,
        age INT NULL,
        gender VARCHAR(16) NULL,
        emotion VARCHAR(32) NULL,
        x INT NULL, y INT NULL, w INT NULL, h INT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """))

# ---------- Video ----------
if args.source == "webcam":
    cap = cv2.VideoCapture(args.device, cv2.CAP_ANY)
else:
    if not args.rtsp_url:
        raise SystemExit("RTSP URL required (set --rtsp-url or RTSP_URL env).")
    # hint: forcing FFMPEG often helps with RTSP
    cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



if not cap.isOpened():
    raise SystemExit("❌ Could not open video source")

print("✅ Video source opened. Press 'q' to quit (if display enabled).")

last_t = 0.0
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("⚠️ No frame received from stream. Re-trying...")
        time.sleep(0.2)
        continue

    now = time.time()
    if now - last_t < args.interval:
        # skip analysis until interval elapsed
        if args.display:
            cv2.imshow("Face Analytics", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    last_t = now

    try:
        analysis = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )
    except Exception as e:
        print("Analysis error:", e)
        continue

    # DeepFace can return dict or list of dicts
    faces = analysis if isinstance(analysis, list) else [analysis] if analysis else []

    # insert into DB
    if faces:
        ts = dt.datetime.utcnow()
        src_label = args.source
        rows = []
        for i, face in enumerate(faces):
            reg = face.get("region", {})
            rows.append({
                "ts": ts,
                "source": src_label,
                "face_index": i,
                "age": int(face.get("age") or 0),
                "gender": str(face.get("dominant_gender") or ""),
                "emotion": str(face.get("dominant_emotion") or ""),
                "x": int(reg.get("x") or 0),
                "y": int(reg.get("y") or 0),
                "w": int(reg.get("w") or 0),
                "h": int(reg.get("h") or 0),
            })

        if rows:
            # bulk insert
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO faces (ts, source, face_index, age, gender, emotion, x, y, w, h)
                        VALUES (:ts, :source, :face_index, :age, :gender, :emotion, :x, :y, :w, :h)
                    """),
                    rows  # SQLAlchemy handles bulk with list of dicts
                )

    # optional overlay
    if args.display:
        for face in faces:
            reg = face.get("region", {})
            x, y, w, h = reg.get('x',0), reg.get('y',0), reg.get('w',0), reg.get('h',0)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            label = f"{face.get('age','?')} | {face.get('dominant_gender','?')} | {face.get('dominant_emotion','?')}"
            cv2.putText(frame, label, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Face Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()