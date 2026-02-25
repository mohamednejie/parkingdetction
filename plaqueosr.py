import warnings
import os
import json
import time
import re

warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

# ==========================================================
# ⚙ CONFIGURATION
# ==========================================================
DEVICE = "cpu"
torch.set_num_threads(4)

MODEL_PATH = "./model/bestplaque2.pt"
IMAGE_PATH = "./data/Aston-Martinjpg.jpg"
RESULT_FOLDER = "./resultplaque"
MAX_IMAGE_SIZE = 1280

print(f"⚙ Device : CPU ({torch.get_num_threads()} threads)\n")

# ==========================================================
# 🔥 CHARGEMENT YOLO (ONNX si dispo)
# ==========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Modèle introuvable")

model = YOLO(MODEL_PATH)

onnx_path = MODEL_PATH.replace(".pt", ".onnx")
if os.path.exists(onnx_path):
    model = YOLO(onnx_path)
    print("✅ ONNX chargé (CPU optimisé)\n")
else:
    print("✅ Modèle PyTorch chargé\n")

# ==========================================================
# 🔤 OCR
# ==========================================================
ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✅ EasyOCR prêt\n")

# ==========================================================
# 🌙 JOUR / NUIT
# ==========================================================
def is_night(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 60

def enhance_night(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ==========================================================
# 🔄 CORRECTION INCLINAISON (STABLE)
# ==========================================================
def deskew_plate(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

        if lines is None:
            return image

        angles = []
        for line in lines[:10]:
            if len(line[0]) == 2:
                rho, theta = line[0]
                angle = (theta - np.pi/2) * 180 / np.pi
                angles.append(angle)

        if len(angles) == 0:
            return image

        median_angle = np.median(angles)

        (h, w) = image.shape[:2]
        center = (w//2, h//2)

        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    except:
        return image

# ==========================================================
# 🧠 PREPROCESSING ROBUSTE
# ==========================================================
def preprocess_plate(plate):

    variants = []

    if plate.shape[0] < 10 or plate.shape[1] < 10:
        return variants

    plate = deskew_plate(plate)

    plate = cv2.resize(plate, None, fx=3, fy=3,
                       interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(3.0, (8,8))
    clahe_img = clahe.apply(gray)
    variants.append(clahe_img)

    _, otsu = cv2.threshold(clahe_img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    variants.append(adaptive)

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    variants.append(morph)

    return variants

# ==========================================================
# 🧹 NETTOYAGE TEXTE
# ==========================================================
def clean_text(text):

    text = text.upper()
    text = ''.join(c for c in text if c.isalnum())

    if len(text) > 10:
        text = text[:10]

    # UK
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{3}', text)
    if match:
        return match.group(0)

    # FR
    match = re.search(r'[A-Z]{2}\d{3}[A-Z]{2}', text)
    if match:
        return match.group(0)

    if 5 <= len(text) <= 10:
        return text

    return ""

# ==========================================================
# 📸 CHARGEMENT IMAGE
# ==========================================================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("❌ Image introuvable")

frame = cv2.imread(IMAGE_PATH)

h, w = frame.shape[:2]
if max(h, w) > MAX_IMAGE_SIZE:
    scale = MAX_IMAGE_SIZE / max(h, w)
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

mode = "NUIT" if is_night(frame) else "JOUR"
frame_used = enhance_night(frame) if mode == "NUIT" else frame.copy()

print(f"📸 Mode : {mode}\n")

# ==========================================================
# 🔍 DETECTION YOLO
# ==========================================================
start_time = time.time()

results = model.predict(
    source=frame_used,
    conf=0.3,
    imgsz=640,
    device="cpu",
    half=False,
    verbose=False
)

detect_time = time.time() - start_time

# ==========================================================
# 📝 TRAITEMENT RESULTATS
# ==========================================================
plates_data = []
plate_id = 0

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_id += 1
        yolo_conf = float(box.conf[0])

        best_text = ""
        best_score = 0

        variants = preprocess_plate(crop)

        for variant in variants:
            try:
                ocr_results = ocr.readtext(variant)

                for det in ocr_results:
                    raw = det[1]
                    ocr_conf = det[2]

                    cleaned = clean_text(raw)
                    score = ocr_conf * yolo_conf

                    if cleaned and score > best_score:
                        best_score = score
                        best_text = cleaned
            except:
                pass

        plates_data.append({
            "id": plate_id,
            "text": best_text,
            "yolo_confidence": round(yolo_conf,3),
            "ocr_confidence": round(best_score,3)
        })

        color = (0,255,0) if best_text else (0,0,255)

        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame,
                    best_text if best_text else "???",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

# ==========================================================
# 💾 SAUVEGARDE
# ==========================================================
os.makedirs(RESULT_FOLDER, exist_ok=True)

result_img = os.path.join(RESULT_FOLDER, "result.jpg")
cv2.imwrite(result_img, frame)

json_path = os.path.join(RESULT_FOLDER, "result.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "mode": mode,
        "device": "CPU",
        "detection_time": round(detect_time,2),
        "plates_detected": len(plates_data),
        "plates": plates_data
    }, f, indent=4)

# ==========================================================
# 📊 RESUME FINAL
# ==========================================================
print("="*50)
print("📊 RÉSUMÉ")
print("="*50)
print(f"Mode              : {mode}")
print(f"Temps détection   : {detect_time:.2f}s")
print(f"Plaques détectées : {len(plates_data)}")

for p in plates_data:
    print(f"#{p['id']} → {p['text']} "
          f"(YOLO:{p['yolo_confidence']:.0%}, "
          f"OCR:{p['ocr_confidence']:.0%})")

print(f"\n📄 Image : {result_img}")
print(f"📄 JSON  : {json_path}")
print("✅ Terminé.")
