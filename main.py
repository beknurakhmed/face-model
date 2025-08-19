import cv2
from deepface import DeepFace
import time

# === CONFIG ===
RTSP_URL = "RTSP URL"
PROCESS_EVERY_N_SECONDS = 1  # analyze every 1 second

# === INIT CAMERA ===
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("❌ Could not connect to RTSP stream")
    exit()

print("✅ Connected to RTSP stream. Press 'q' to quit.")

last_process_time = 0
analysis_result = None

while True:
    ret, frame = cap.read()
    analysis = DeepFace.analyze(
    frame,
    actions=['age', 'gender', 'emotion'],
    enforce_detection=False
    )
    print(analysis)
    if not ret:
        print("⚠️ No frame received from stream")
        break

    now = time.time()
    if now - last_process_time >= PROCESS_EVERY_N_SECONDS:
        try:
            analysis_result = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )
            last_process_time = now
        except Exception as e:
            print("Analysis error:", e)
            analysis_result = None

    # === DRAW RESULTS ===
    if analysis_result is not None:
        if isinstance(analysis_result, list):
            # Multiple faces
            for face in analysis_result:
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                age = face["age"]
                gender = face["gender"]
                emotion = face["dominant_emotion"]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                # Put text
                label = f"{age} | {gender} | {emotion}"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            # Single face result
            age = analysis_result["age"]
            gender = analysis_result["gender"]
            emotion = analysis_result["dominant_emotion"]

            cv2.putText(frame, f"Age:{age} Gender:{gender} Emotion:{emotion}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # === SHOW STREAM ===
    cv2.imshow("RTSP Face Model", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
