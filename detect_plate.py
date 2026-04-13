import os
import cv2
import numpy as np
import torch
import warnings
import re
from ultralytics import YOLO
import easyocr

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ==========================================
# CHARGEMENT DES MODÈLES (Une seule fois)
# ==========================================
print("🔄 Chargement du modèle YOLO...")
model_plate = YOLO("model/bestplaque2.pt")

print("🔄 Chargement de EasyOCR...")
ocr = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)

def deskew_plate(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        if lines is None: return image
        angles = []
        for line in lines[:10]:
            if len(line[0]) == 2:
                rho, theta = line[0]
                angles.append((theta - np.pi / 2) * 180 / np.pi)
        if not angles: return image
        median_angle = np.median(angles)
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except:
        return image

def expand_bbox(x1, y1, x2, y2, img_w, img_h, margin=0.25):
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin * 0.5)
    return max(0, x1 - mx), max(0, y1 - my), min(img_w, x2 + mx), min(img_h, y2 + my)

def clean_text(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    if len(text) > 12: text = text[:12]
    match = re.search(r'[NM][CG](\d{4,6})', text)
    if match: return f"NC{match.group(1).zfill(6)}"
    return text if len(text) >= 4 else ""

def read_license_plate(image_path):
    """ Analyse une image sauvegardée et retourne le texte de la plaque """
    frame = cv2.imread(image_path)
    if frame is None:
        return None, 0.0, "Image illisible"

    # Amélioration si image très sombre
    if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 60:
        clahe = cv2.createCLAHE(3.0, (8, 8))
        frame = cv2.cvtColor(clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

    # 1️⃣ ON BAISSE LA CONFIANCE À 0.15 ICI
    results = model_plate.predict(source=frame, conf=0.15, imgsz=640, device="cpu", verbose=False)

    best_text = ""
    best_score = 0.0

    # 2️⃣ DEBUG : YOLO A-T-IL PERDU LA PLAQUE ?
    if len(results[0].boxes) == 0:
        print("   ❌ [DEBUG] YOLO a vu la plaque en direct, mais l'a perdue sur la photo sauvegardée !")

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])

            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            crop = frame[y1e:y2e, x1e:x2e]
            if crop.size == 0: continue

            crop = deskew_plate(crop)
            
            ocr_results = ocr.readtext(crop, detail=1, paragraph=False, min_size=10, text_threshold=0.5)
            full_text = ""
            total_conf = 0
            n_parts = 0

            for det in ocr_results:
                full_text += det[1]
                total_conf += det[2]
                n_parts += 1

            # 3️⃣ DEBUG : QUE LIT VRAIMENT L'OCR ?
            if full_text:
                print(f"   🧐 [DEBUG OCR] Texte brut lu par l'IA : '{full_text}'")

            if n_parts > 0:
                cleaned = clean_text(full_text)
                
                # 4️⃣ DEBUG : QUE RESTE-T-IL APRÈS NETTOYAGE ?
                print(f"   🧹 [DEBUG NETTOYAGE] Résultat après filtrage : '{cleaned}'")

                score = (total_conf / n_parts) * yolo_conf
                if cleaned and score > best_score:
                    best_score = score
                    best_text = cleaned

    if best_text:
        return best_text, round(best_score, 3), "Succès"
    else:
        print("   🗑️ [DEBUG] La plaque a été ignorée (trop courte ou illisible).")
        return None, 0.0, "Plaque illisible"