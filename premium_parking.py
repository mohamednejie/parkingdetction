import cv2
import time
import math
import os
import requests
import numpy as np
from ultralytics import YOLO

class PremiumParkingTracker:
    def __init__(self, model_path="model/yolo26n.pt"):
        print("🚀 Initialisation du Tracker Premium...")
        self.model = YOLO(model_path)
        
        self.tracked_cars  = {}
        self.illegal_start = {}
        self.captured      = set()
        
        self.ILLEGAL_TIME = 10
        self.CAPTURE_DIR  = "captures_parking/infractions"
        os.makedirs(self.CAPTURE_DIR, exist_ok=True)

        self.COLOR_FREE     = (0, 200, 0)
        self.COLOR_OCCUPIED = (255, 150, 0)
        self.COLOR_ILLEGAL  = (0, 0, 255)
        self.COLOR_TEXT     = (255, 255, 255)

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

    def get_car_id(self, box, max_dist=50):
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        for car_id, (tx, ty) in self.tracked_cars.items():
            if math.hypot(cx - tx, cy - ty) < max_dist:
                self.tracked_cars[car_id] = (cx, cy)
                return car_id
        new_id = len(self.tracked_cars) + 1
        self.tracked_cars[new_id] = (cx, cy)
        return new_id

    def draw_dashboard(self, frame, free, occupied, illegal):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 140), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "PREMIUM PARKING SYSTEM", (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT,     2)
        cv2.putText(frame, f"Places Libres : {free}",   (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_FREE,    2)
        cv2.putText(frame, f"Occupees      : {occupied}",(20, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_OCCUPIED,2)
        cv2.putText(frame, f"Infractions   : {illegal}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_ILLEGAL, 2)

    def process_frame(self, frame, parking_slots, webhook_url=None):
        """
        Analyse le frame avec les polygones fournis.
        
        ✅ SANS resize fixe : on travaille à la résolution native de la caméra.
        Les slots doivent être fournis en pixels absolus correspondant
        à la taille réelle du frame (H × W de la caméra).
        La conversion relative→pixels est faite dans app.py (zone_worker)
        en utilisant les dimensions réelles du frame.
        """
        # ✅ Suppression du cv2.resize — on garde la résolution native
        # frame = cv2.resize(frame, (1250, 600))  ← SUPPRIMÉ

        results = self.model.predict(source=frame, conf=0.30, device="cpu", verbose=False)[0]

        current_time = time.time()
        cars = []

        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cars.append((x1, y1, x2, y2, cx, cy))

        free_count     = 0
        occupied_count = 0

        for slot in parking_slots:
            occupied = any(self.point_in_polygon((cx, cy), slot) for _, _, _, _, cx, cy in cars)
            color    = self.COLOR_OCCUPIED if occupied else self.COLOR_FREE
            label    = "OCCUPEE" if occupied else "LIBRE"
            cv2.polylines(frame, [np.array(slot, dtype=np.int32)], True, color, 2)
            cv2.putText(frame, label, tuple(slot[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if occupied: occupied_count += 1
            else:        free_count     += 1

        for x1, y1, x2, y2, cx, cy in cars:
            car_id  = self.get_car_id([x1, y1, x2, y2])
            in_slot = any(self.point_in_polygon((cx, cy), s) for s in parking_slots)

            if in_slot:
                self.illegal_start.pop(car_id, None)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_OCCUPIED, 2)
                continue

            if car_id not in self.illegal_start:
                self.illegal_start[car_id] = current_time

            duration = current_time - self.illegal_start[car_id]

            if duration >= self.ILLEGAL_TIME:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_ILLEGAL, 3)
                cv2.putText(frame, f"INFRACTION ({int(duration)}s)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.COLOR_ILLEGAL, 2)

                if car_id not in self.captured:
                    filename = f"{self.CAPTURE_DIR}/infraction_{car_id}_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    self.captured.add(car_id)
                    if webhook_url:
                        try:
                            requests.post(webhook_url, json={"alert": "infraction", "photo": filename}, timeout=2)
                        except: pass

        self.draw_dashboard(frame, free_count, occupied_count, len(self.captured))
        return frame, free_count, occupied_count, len(self.captured)