import cv2
import json
import time
import os
import math
import numpy as np
from ultralytics import YOLO

# ==================================================
# CONFIGURATION
# ==================================================
VIDEO_PATH = "data/parking1.mp4"
MODEL_PATH = "./model/yolo26n.pt"
JSON_PATH = "bounding_boxes.json"

TARGET_W, TARGET_H = 1250, 600
ILLEGAL_TIME = 10  # secondes
CAPTURE_DIR = "captures"

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ==================================================
# COULEURS (BGR)
# ==================================================
COLOR_FREE = (0, 200, 0)        # Vert
COLOR_OCCUPIED = (255, 150, 0)  # Bleu
COLOR_ILLEGAL = (0, 0, 255)     # Rouge
COLOR_SLOT = (0, 255, 255)      # Jaune
COLOR_TEXT = (255, 255, 255)

# ==================================================
# CHARGER LES PLACES (POLYGONES)
# ==================================================
def load_parking_slots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    slots = []
    for item in data:
        slots.append([(int(x), int(y)) for x, y in item["points"]])
    return slots

# ==================================================
# TEST POINT DANS POLYGONE
# ==================================================
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(
        np.array(polygon, dtype=np.int32),
        point,
        False
    ) >= 0

# ==================================================
# TRACKING SIMPLE (DISTANCE)
# ==================================================
def get_car_id(box, tracked, max_dist=50):
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    for car_id, (tx, ty) in tracked.items():
        if math.hypot(cx - tx, cy - ty) < max_dist:
            tracked[car_id] = (cx, cy)
            return car_id

    new_id = len(tracked) + 1
    tracked[new_id] = (cx, cy)
    return new_id

# ==================================================
# DASHBOARD PROFESSIONNEL
# ==================================================
def draw_dashboard(frame, free, occupied, illegal):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 140), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "SMART PARKING SYSTEM", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

    cv2.putText(frame, f"Libres      : {free}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FREE, 2)

    cv2.putText(frame, f"Occupees    : {occupied}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_OCCUPIED, 2)

    cv2.putText(frame, f"Infractions : {illegal}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ILLEGAL, 2)

# ==================================================
# INITIALISATION
# ==================================================
model = YOLO(MODEL_PATH)
parking_slots = load_parking_slots(JSON_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "❌ Impossible d'ouvrir la vidéo"

fps = int(cap.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(
    "parking_management_final.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (TARGET_W, TARGET_H)
)

tracked_cars = {}
illegal_start = {}
captured = set()

frame_count = 0

# ==================================================
# BOUCLE PRINCIPALE
# ==================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    results = model(frame)[0]

    current_time = time.time()

    # -----------------------------
    # Récupération des voitures
    # -----------------------------
    cars = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 2:  # voiture
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cars.append((x1, y1, x2, y2, cx, cy))

    # -----------------------------
    # PLACES LIBRES / OCCUPEES
    # -----------------------------
    free_count = 0
    occupied_count = 0

    for slot in parking_slots:
        occupied = any(point_in_polygon((cx, cy), slot) for _, _, _, _, cx, cy in cars)

        color = COLOR_OCCUPIED if occupied else COLOR_FREE
        label = "OCCUPEE" if occupied else "LIBRE"

        cv2.polylines(frame, [np.array(slot)], True, color, 2)
        cv2.putText(frame, label, slot[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if occupied:
            occupied_count += 1
        else:
            free_count += 1

    # -----------------------------
    # MAL GAREE + TIMER
    # -----------------------------
    for x1, y1, x2, y2, cx, cy in cars:
        car_id = get_car_id([x1, y1, x2, y2], tracked_cars)

        in_slot = any(point_in_polygon((cx, cy), s) for s in parking_slots)

        if in_slot:
            illegal_start.pop(car_id, None)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OCCUPIED, 2)
            continue

        if car_id not in illegal_start:
            illegal_start[car_id] = current_time

        duration = current_time - illegal_start[car_id]

        if duration >= ILLEGAL_TIME:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ILLEGAL, 3)
            cv2.putText(frame,
                        f"INFRACTION ({int(duration)}s)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        COLOR_ILLEGAL,
                        2)

            if car_id not in captured:
                filename = f"{CAPTURE_DIR}/infraction_{car_id}.jpg"
                cv2.imwrite(filename, frame)
                captured.add(car_id)
                print(f"📸 Capture sauvegardée : {filename}")

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    draw_dashboard(frame, free_count, occupied_count, len(captured))

    video_writer.write(frame)
    cv2.imshow("Smart Parking Management", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==================================================
# FIN
# ==================================================
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"✅ Terminé – {frame_count} frames traitées")
