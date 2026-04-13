import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# CONFIGURATION & MODÈLE
# ==========================================
print("🚀 Chargement du modèle Basic Parking (yolo26s.pt)...")
# Note: Remplacez par le bon chemin si nécessaire
model_basic = YOLO("./model/yolo26s.pt") 
TARGET_CLASSES = [2]  # Voitures (Classe 2 dans COCO)

def detect_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        return 'NUIT', brightness
    elif brightness < 100:
        return 'CREPUSCULE', brightness
    return 'JOUR', brightness

def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)

def enhance_twilight(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def get_conf(mode):
    if mode == 'NUIT': return 0.15
    elif mode == 'CREPUSCULE': return 0.20
    return 0.25

def draw_overlay(frame, boxes, mode, brightness, car_count):
    """ Votre superbe fonction d'interface, allégée pour du Live (sans timeline vidéo) """
    fh, fw = frame.shape[:2]

    # ─── Dessiner les voitures ───
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            colors = {
                'JOUR':       (0, 255, 0),
                'CREPUSCULE': (0, 200, 255),
                'NUIT':       (0, 255, 255)
            }
            color = colors.get(mode, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            label = f"#{i+1} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ─── Bandeau haut ───
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # ─── LIVE indicator ───
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (20, 20), 6, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Heure actuelle
    current_time = time.strftime("%H:%M:%S")
    cv2.putText(frame, current_time, (85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ─── Mode jour/nuit ───
    mode_config = {
        'JOUR':       {"color": (0, 255, 255)},
        'CREPUSCULE': {"color": (0, 180, 255)},
        'NUIT':       {"color": (255, 200, 100)}
    }
    cfg = mode_config.get(mode, mode_config['JOUR'])

    mode_x = 220
    cv2.circle(frame, (mode_x, 20), 12, cfg['color'], -1)
    cv2.circle(frame, (mode_x, 20), 12, (255, 255, 255), 1)
    cv2.putText(frame, mode, (mode_x + 18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg['color'], 2)

    # ─── Compteur principal ───
    cv2.putText(frame, f"VOITURES : {car_count}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ─── Infos ───
    info = f"Lum: {brightness:.0f}/255 | IA: YOLOv8s"
    cv2.putText(frame, info, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ─── Compteur coin droit ───
    box_x = fw - 110
    box_y = 20
    cv2.rectangle(frame, (box_x, box_y), (box_x + 90, box_y + 55), (0, 80, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + 90, box_y + 55), (0, 255, 0), 2)
    cv2.putText(frame, f"{car_count}", (box_x + 15, box_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    return frame

def process_basic_parking_frame(frame):
    """ Reçoit une image de la caméra, l'analyse et retourne le compte et l'image dessinée """
    mode, brightness = detect_condition(frame)

    if mode == 'NUIT':
        detect_frame = enhance_night(frame)
    elif mode == 'CREPUSCULE':
        detect_frame = enhance_twilight(frame)
    else:
        detect_frame = frame

    conf = get_conf(mode)
    results = model_basic.predict(source=detect_frame, classes=TARGET_CLASSES, conf=conf, imgsz=640, verbose=False)

    boxes = results[0].boxes
    car_count = len(boxes) if boxes is not None else 0

    annotated_frame = draw_overlay(frame.copy(), boxes, mode, brightness, car_count)
    
    return car_count, annotated_frame