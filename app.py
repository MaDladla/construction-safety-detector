import os
import uuid
import subprocess
from collections import Counter

import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONF = float(os.environ.get("CONF_THRESHOLD", 0.4))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
model = YOLO(os.path.join(BASE_DIR, "best.pt"))


def overlaps(worker, item):
    """True if the centre of an equipment box lies inside a worker box."""
    cx = (item[0] + item[2]) / 2
    cy = (item[1] + item[3]) / 2
    return worker[0] <= cx <= worker[2] and worker[1] <= cy <= worker[3]


def check_frame(boxes, names):
    """Return number of workers missing a helmet or vest in one frame."""
    by_class = {}
    for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
        by_class.setdefault(names[int(cls)], []).append(box)
    workers = by_class.get("worker", [])
    no_helmet = sum(not any(overlaps(w, h) for h in by_class.get("helmet", [])) for w in workers)
    no_vest = sum(not any(overlaps(w, v) for v in by_class.get("vest", [])) for w in workers)
    return len(workers), no_helmet, no_vest


def process_video(src, dst):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst + ".raw.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts = Counter()
    frames = max_workers = max_no_helmet = max_no_vest = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=CONF, verbose=False)[0]
        names = result.names
        for cls in result.boxes.cls.tolist():
            counts[names[int(cls)]] += 1
        workers, no_helmet, no_vest = check_frame(result.boxes, names)
        max_workers = max(max_workers, workers)
        max_no_helmet = max(max_no_helmet, no_helmet)
        max_no_vest = max(max_no_vest, no_vest)

        annotated = result.plot()
        if no_helmet or no_vest:
            cv2.putText(annotated, f"WARNING: {no_helmet} no helmet, {no_vest} no vest",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(annotated)
        frames += 1
    cap.release()
    writer.release()

    # Re-encode to H.264 so browsers can play the result
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
    os.remove(tmp)

    return {
        "frames": frames,
        "detections": dict(counts),
        "max_workers": max_workers,
        "max_no_helmet": max_no_helmet,
        "max_no_vest": max_no_vest,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("video")
    if not file or not file.filename:
        return render_template("index.html", error="Please choose a video file.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return render_template("index.html", error=f"Unsupported file type: {ext}")

    job = uuid.uuid4().hex
    src = os.path.join(UPLOAD_DIR, job + ext)
    out_name = job + ".mp4"
    file.save(src)
    try:
        stats = process_video(src, os.path.join(OUTPUT_DIR, out_name))
    except Exception as exc:
        return render_template("index.html", error=f"Processing failed: {exc}")
    finally:
        if os.path.exists(src):
            os.remove(src)
    return render_template("index.html", video=out_name, stats=stats)


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
