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
VIDEO_PATH = "./data/plaquevedio.mp4"
RESULT_FOLDER = "./resultplaque"
MAX_IMAGE_SIZE = 1280

# 🎬 PARAMÈTRES VIDÉO
FRAME_SKIP = 5
SAVE_VIDEO = True
SAVE_FRAMES = True
SAVE_CROPS = True
MAX_FRAMES = None

# ==========================================================
# 🎯 CONFIGURATION MATCHING
# ==========================================================
MATCH_THRESHOLD = 0.5   # Score minimum pour valider

# Base de données des plaques autorisées
AUTHORIZED_PLATES = [
    "NC128883",
    "NC245671",
    "NC331200",
    "NC987654",
    "AB123CD",
    "EF456GH",
]

# Ou charger depuis fichier JSON
PLATES_DB_FILE = "./data/plates_db.json"

print(f"⚙ Device : CPU ({torch.get_num_threads()} threads)")
print(f"⚙ Seuil matching : {MATCH_THRESHOLD:.0%}\n")

# ==========================================================
# 📂 CHARGEMENT BASE DE DONNÉES PLAQUES
# ==========================================================
def load_plates_database():
    """Charge les plaques depuis liste + fichier JSON"""
    plates = set(p.upper().strip() for p in AUTHORIZED_PLATES)

    if os.path.exists(PLATES_DB_FILE):
        try:
            with open(PLATES_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for p in data:
                    plates.add(p.upper().strip())
            elif isinstance(data, dict) and "plates" in data:
                for p in data["plates"]:
                    plates.add(p.upper().strip())
            print(f"📂 Base chargée depuis {PLATES_DB_FILE}")
        except Exception as e:
            print(f"⚠ Fichier DB non chargé : {e}")

    return plates

plates_db = load_plates_database()
print(f"📋 Plaques en base : {len(plates_db)}")
for p in sorted(plates_db):
    print(f"    • {p}")
print()

# ==========================================================
# 🎯 FONCTION DE MATCHING / VÉRIFICATION
# ==========================================================
def levenshtein_distance(s1, s2):
    """Calcule la distance d'édition entre 2 chaînes"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,      # suppression
                curr[j] + 1,           # insertion
                prev[j] + (c1 != c2)   # substitution
            ))
        prev = curr
    return prev[-1]


def calculate_similarity(detected, reference):
    """
    Calcule un score de similarité entre 0.0 et 1.0

    Combine 3 méthodes :
      - Position par position (50%)
      - Distance de Levenshtein (35%)
      - Caractères communs (15%)
    """
    if not detected or not reference:
        return 0.0

    d = detected.upper().strip()
    r = reference.upper().strip()

    # Match exact
    if d == r:
        return 1.0

    # Contenu l'un dans l'autre
    if d in r or r in d:
        shorter = min(len(d), len(r))
        longer = max(len(d), len(r))
        return shorter / longer

    max_len = max(len(d), len(r))
    min_len = min(len(d), len(r))

    # Méthode 1 : Position par position
    pos_match = sum(1 for i in range(min_len) if d[i] == r[i])
    pos_score = pos_match / max_len

    # Méthode 2 : Levenshtein
    lev = levenshtein_distance(d, r)
    lev_score = 1.0 - (lev / max_len)

    # Méthode 3 : Caractères communs
    from collections import Counter
    common = sum((Counter(d) & Counter(r)).values())
    char_score = (2 * common) / (len(d) + len(r))

    # Score pondéré
    score = (0.50 * pos_score +
             0.35 * lev_score +
             0.15 * char_score)

    # Bonus même préfixe (ex: NC)
    if len(d) >= 2 and len(r) >= 2 and d[:2] == r[:2]:
        score = min(1.0, score + 0.05)

    # Bonus même longueur
    if len(d) == len(r):
        score = min(1.0, score + 0.03)

    return round(score, 4)


def match_plate(detected_text, database, threshold=MATCH_THRESHOLD):
    """
    Vérifie si la plaque détectée correspond à une plaque
    dans la base de données.

    Returns:
        dict {
            detected, matched, best_match, score,
            status, all_scores
        }
    """
    if not detected_text or not database:
        return {
            "detected": detected_text or "",
            "matched": False,
            "best_match": None,
            "score": 0.0,
            "status": "NO_DB"
        }

    best_match = None
    best_score = 0.0
    all_scores = {}

    for db_plate in database:
        score = calculate_similarity(detected_text, db_plate)
        all_scores[db_plate] = round(score, 3)

        if score > best_score:
            best_score = score
            best_match = db_plate

    matched = best_score >= threshold

    if best_score >= 0.95:
        status = "EXACT"
    elif best_score >= threshold:
        status = "MATCH"
    elif best_score >= 0.5:
        status = "PARTIAL"
    else:
        status = "NO_MATCH"

    return {
        "detected": detected_text,
        "matched": matched,
        "best_match": best_match,
        "score": round(best_score, 4),
        "status": status,
        "all_scores": dict(sorted(all_scores.items(),
                                   key=lambda x: x[1],
                                   reverse=True)[:5])
    }


def get_match_color(status):
    """Couleur BGR selon le statut"""
    return {
        "EXACT":    (0, 255, 0),      # Vert
        "MATCH":    (0, 200, 100),    # Vert clair
        "PARTIAL":  (0, 165, 255),    # Orange
        "NO_MATCH": (0, 0, 255),      # Rouge
        "NO_DB":    (128, 128, 128),  # Gris
    }.get(status, (255, 255, 255))


def get_match_emoji(status):
    return {
        "EXACT":    "✅",
        "MATCH":    "🟢",
        "PARTIAL":  "🟡",
        "NO_MATCH": "🔴",
        "NO_DB":    "⚪",
    }.get(status, "❓")

# ==========================================================
# 🔥 CHARGEMENT YOLO
# ==========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Modèle introuvable")

model = YOLO(MODEL_PATH)

onnx_path = MODEL_PATH.replace(".pt", ".onnx")
if os.path.exists(onnx_path):
    model = YOLO(onnx_path)
    print("✅ ONNX chargé\n")
else:
    print("✅ Modèle PyTorch chargé\n")

# ==========================================================
# 🔤 OCR
# ==========================================================
ocr = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)
print("✅ EasyOCR prêt (EN + FR)\n")

# ==========================================================
# 🌙 JOUR / NUIT
# ==========================================================
def is_night(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 60

def enhance_night(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ==========================================================
# 🔄 CORRECTION INCLINAISON
# ==========================================================
def deskew_plate(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        if lines is None:
            return image
        angles = []
        for line in lines[:10]:
            if len(line[0]) == 2:
                rho, theta = line[0]
                angle = (theta - np.pi / 2) * 180 / np.pi
                angles.append(angle)
        if len(angles) == 0:
            return image
        median_angle = np.median(angles)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except:
        return image

# ==========================================================
# 🔲 EXPANSION DU CROP
# ==========================================================
def expand_bbox(x1, y1, x2, y2, img_w, img_h, margin=0.25):
    w = x2 - x1
    h = y2 - y1
    mx = int(w * margin)
    my = int(h * margin * 0.5)
    x1_new = max(0, x1 - mx)
    y1_new = max(0, y1 - my)
    x2_new = min(img_w, x2 + mx)
    y2_new = min(img_h, y2 + my)
    return x1_new, y1_new, x2_new, y2_new

# ==========================================================
# 🧠 PREPROCESSING
# ==========================================================
def preprocess_plate(plate):
    variants = []
    if plate.shape[0] < 10 or plate.shape[1] < 10:
        return variants

    plate = deskew_plate(plate)
    scale = max(3, 100 / plate.shape[0])
    scale = min(scale, 6)
    plate = cv2.resize(plate, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    pad = 20
    plate = cv2.copyMakeBorder(plate, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(3.0, (8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(clahe_img)

    _, otsu = cv2.threshold(clahe_img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    _, otsu_inv = cv2.threshold(clahe_img, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    variants.append(adaptive)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    variants.append(morph)

    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(clahe_img, -1, sharpen_kernel)
    variants.append(sharpened)

    variants.append(plate)

    return variants

# ==========================================================
# 🧹 NETTOYAGE TEXTE
# ==========================================================
CHAR_FIXES = {
    'O': '0', 'Q': '0', 'D': '0',
    '0': 'O', '8': 'B',
    '1': 'I', '5': 'S',
    '|': 'I', '!': 'I',
}

def fix_plate_format(text):
    text = text.upper().strip()
    text = text.replace(' ', '').replace('-', '').replace('.', '')
    text = ''.join(c for c in text if c.isalnum())
    if len(text) > 12:
        text = text[:12]

    match = re.search(r'[NM][CG](\d{4,6})', text)
    if match:
        digits = match.group(1).zfill(6)
        return f"NC{digits}"

    match = re.search(r'(\d{6,8})', text)
    if match:
        digits = match.group(1)
        idx = text.find(digits)
        prefix = text[:idx] if idx > 0 else ""
        if prefix:
            prefix = prefix.replace('0', 'O').replace('8', 'B')
            return f"{prefix}{digits[:6]}"
        else:
            return f"NC{digits[:6]}"

    match = re.search(r'([A-Z]{2})(\d{3})([A-Z]{2})', text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{3})', text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    if len(text) >= 5:
        return text
    return ""

def clean_text(text):
    if not text or len(text.strip()) < 2:
        return ""
    cleaned = fix_plate_format(text)
    return cleaned

# ==========================================================
# 🎬 OUVERTURE VIDÉO
# ==========================================================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"❌ Vidéo introuvable : {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Impossible d'ouvrir la vidéo")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

print("=" * 50)
print("🎬 INFOS VIDÉO")
print("=" * 50)
print(f"  Fichier    : {VIDEO_PATH}")
print(f"  Résolution : {width}x{height}")
print(f"  FPS        : {fps}")
print(f"  Frames     : {total_frames}")
print(f"  Durée      : {duration:.1f}s")
print(f"  Frame skip : 1/{FRAME_SKIP}")
print("=" * 50 + "\n")

# ==========================================================
# 💾 PRÉPARATION SORTIE
# ==========================================================
os.makedirs(RESULT_FOLDER, exist_ok=True)

if SAVE_FRAMES:
    frames_folder = os.path.join(RESULT_FOLDER, "frames")
    os.makedirs(frames_folder, exist_ok=True)

if SAVE_CROPS:
    crops_folder = os.path.join(RESULT_FOLDER, "crops")
    os.makedirs(crops_folder, exist_ok=True)

scale = 1.0
if max(height, width) > MAX_IMAGE_SIZE:
    scale = MAX_IMAGE_SIZE / max(height, width)
    out_w = int(width * scale)
    out_h = int(height * scale)
else:
    out_w, out_h = width, height

writer = None
if SAVE_VIDEO:
    output_video = os.path.join(RESULT_FOLDER, "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = max(1, fps // FRAME_SKIP)
    writer = cv2.VideoWriter(output_video, fourcc, out_fps,
                              (out_w, out_h))

# ==========================================================
# 🔁 BOUCLE PRINCIPALE
# ==========================================================
all_plates = {}
all_detections = []
all_match_results = []       # ← NOUVEAU : résultats matching
frame_count = 0
processed_count = 0
crop_id = 0
total_start = time.time()

print("🚀 Traitement en cours...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if MAX_FRAMES and frame_count > MAX_FRAMES:
        break

    if frame_count % FRAME_SKIP != 0:
        continue

    processed_count += 1

    if scale != 1.0:
        frame = cv2.resize(frame, (out_w, out_h))

    night = is_night(frame)
    frame_used = enhance_night(frame) if night else frame.copy()

    # ── Détection YOLO ──
    t0 = time.time()
    results = model.predict(
        source=frame_used,
        conf=0.25,
        imgsz=640,
        device="cpu",
        half=False,
        verbose=False
    )
    detect_time = time.time() - t0

    frame_plates = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1e, y1e, x2e, y2e = expand_bbox(
                x1, y1, x2, y2,
                frame.shape[1], frame.shape[0],
                margin=0.25
            )

            crop = frame[y1e:y2e, x1e:x2e]
            if crop.size == 0:
                continue

            crop_id += 1
            yolo_conf = float(box.conf[0])

            if SAVE_CROPS:
                cv2.imwrite(
                    os.path.join(crops_folder,
                                 f"crop_{crop_id:04d}_f{frame_count}.jpg"),
                    crop
                )

            # ── OCR multi-variantes ──
            best_text = ""
            best_score = 0
            all_raw = []

            variants = preprocess_plate(crop)

            for vi, variant in enumerate(variants):
                try:
                    ocr_results = ocr.readtext(
                        variant, detail=1, paragraph=False,
                        min_size=10, text_threshold=0.5,
                        low_text=0.3, width_ths=1.0,
                        add_margin=0.15
                    )

                    full_text = ""
                    total_conf = 0
                    n_parts = 0

                    for det in ocr_results:
                        raw = det[1]
                        ocr_conf = det[2]
                        all_raw.append(
                            f"v{vi}:{raw}({ocr_conf:.2f})"
                        )
                        full_text += raw
                        total_conf += ocr_conf
                        n_parts += 1

                    if n_parts > 0:
                        avg_conf = total_conf / n_parts
                        cleaned = clean_text(full_text)
                        score = avg_conf * yolo_conf

                        if cleaned and score > best_score:
                            best_score = score
                            best_text = cleaned

                    for det in ocr_results:
                        raw = det[1]
                        ocr_conf = det[2]
                        cleaned = clean_text(raw)
                        score = ocr_conf * yolo_conf

                        if cleaned and len(cleaned) > len(best_text) \
                           and score > best_score * 0.8:
                            best_score = score
                            best_text = cleaned

                except Exception as e:
                    pass

            # ══════════════════════════════════════════
            # 🎯 MATCHING / VÉRIFICATION
            # ══════════════════════════════════════════
            match_result = match_plate(
                best_text, plates_db, MATCH_THRESHOLD
            )

            # ── Log avec résultat matching ──
            if all_raw:
                emoji = get_match_emoji(match_result["status"])
                match_info = ""
                if match_result["matched"]:
                    match_info = (
                        f" → {emoji} MATCH: "
                        f"{match_result['best_match']} "
                        f"({match_result['score']:.0%})"
                    )
                elif best_text:
                    match_info = (
                        f" → {emoji} {match_result['status']} "
                        f"(best: {match_result['best_match']} "
                        f"{match_result['score']:.0%})"
                    )

                print(f"    🔍 Crop#{crop_id} F{frame_count} | "
                      f"Raw: {all_raw[:3]} → {best_text}"
                      f"{match_info}")

            # ── Stocker ──
            if best_text:
                if best_text not in all_plates or \
                   best_score > all_plates[best_text]:
                    all_plates[best_text] = best_score

                frame_plates.append(best_text)

                detection_entry = {
                    "frame": frame_count,
                    "time_s": round(frame_count / fps, 2),
                    "text": best_text,
                    "yolo_conf": round(yolo_conf, 3),
                    "score": round(best_score, 3),
                    "bbox": [x1, y1, x2, y2],
                    "bbox_expanded": [x1e, y1e, x2e, y2e],
                    "match": match_result     # ← AJOUT
                }
                all_detections.append(detection_entry)

                # Stocker les résultats matching
                all_match_results.append({
                    "frame": frame_count,
                    "time_s": round(frame_count / fps, 2),
                    **match_result
                })

            # ══════════════════════════════════════════
            # 🎨 DESSINER AVEC COULEUR DU MATCH
            # ══════════════════════════════════════════
            if best_text:
                color = get_match_color(match_result["status"])

                # Rectangle épais
                cv2.rectangle(frame, (x1e, y1e), (x2e, y2e),
                              color, 3)

                # Label avec score matching
                if match_result["matched"]:
                    label = (f"{best_text} "
                             f"[{match_result['score']:.0%}]")
                else:
                    label = f"{best_text} [?]"

                # Fond coloré pour le label
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x1e, y1e - th - 15),
                    (x1e + tw + 10, y1e),
                    color, -1
                )
                cv2.putText(
                    frame, label,
                    (x1e + 5, y1e - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                # Statut sous la bbox
                cv2.putText(
                    frame, match_result["status"],
                    (x1e, y2e + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
            else:
                # Pas de texte détecté
                cv2.rectangle(frame, (x1e, y1e), (x2e, y2e),
                              (0, 0, 255), 2)
                cv2.putText(frame, "???",
                            (x1e, y1e - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

    # ── Info frame ──
    info = (f"F{frame_count}/{total_frames} | "
            f"Plaques: {len(all_plates)} | "
            f"{detect_time:.2f}s")
    cv2.putText(frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    # ── Légende matching en bas de frame ──
    y_leg = frame.shape[0] - 80
    cv2.putText(frame, "EXACT/MATCH (>=70%)",
                (10, y_leg),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1)
    cv2.putText(frame, "PARTIAL (50-70%)",
                (10, y_leg + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 165, 255), 1)
    cv2.putText(frame, "NO MATCH (<50%)",
                (10, y_leg + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1)

    if SAVE_FRAMES and frame_plates:
        fname = (f"frame_{frame_count:06d}_"
                 f"{'_'.join(frame_plates)}.jpg")
        cv2.imwrite(os.path.join(frames_folder, fname), frame)

    if writer:
        writer.write(frame)

    if processed_count % 10 == 0:
        elapsed = time.time() - total_start
        progress = frame_count / total_frames * 100
        print(f"  ⏳ {progress:5.1f}% | "
              f"F{frame_count}/{total_frames} | "
              f"Uniques: {len(all_plates)} | "
              f"{elapsed:.0f}s")

# ==========================================================
# 🔗 POST-TRAITEMENT : FUSION PLAQUES SIMILAIRES
# ==========================================================
def merge_similar_plates(plates_dict, min_similarity=0.7):
    texts = list(plates_dict.keys())
    merged = {}
    used = set()
    texts.sort(key=len, reverse=True)

    for t1 in texts:
        if t1 in used:
            continue
        best_key = t1
        best_score = plates_dict[t1]

        for t2 in texts:
            if t2 == t1 or t2 in used:
                continue
            if t2 in t1 or t1 in t2:
                used.add(t2)
                if plates_dict[t2] > best_score:
                    best_score = plates_dict[t2]
                continue

            common = sum(1 for c in t2 if c in t1)
            sim = common / max(len(t1), len(t2))
            if sim > min_similarity:
                used.add(t2)
                if plates_dict[t2] > best_score:
                    best_score = plates_dict[t2]

        merged[best_key] = best_score
        used.add(t1)

    return merged

all_plates = merge_similar_plates(all_plates)

# ==========================================================
# 🎯 MATCHING FINAL SUR PLAQUES UNIQUES
# ==========================================================
final_match_results = []

for text, ocr_score in sorted(all_plates.items(),
                                key=lambda x: x[1],
                                reverse=True):
    mr = match_plate(text, plates_db, MATCH_THRESHOLD)
    final_match_results.append({
        "plate": text,
        "ocr_score": round(ocr_score, 3),
        **mr
    })

# ==========================================================
# 🧹 NETTOYAGE
# ==========================================================
cap.release()
if writer:
    writer.release()

total_time = time.time() - total_start

# ==========================================================
# 💾 SAUVEGARDE JSON
# ==========================================================
unique_plates = []
for text, score in sorted(all_plates.items(),
                           key=lambda x: x[1],
                           reverse=True):
    mr = match_plate(text, plates_db, MATCH_THRESHOLD)
    unique_plates.append({
        "text": text,
        "best_score": round(score, 3),
        "match": mr
    })

matched_count = sum(1 for p in unique_plates
                    if p["match"]["matched"])

json_path = os.path.join(RESULT_FOLDER, "result.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "video": VIDEO_PATH,
        "device": "CPU",
        "match_threshold": MATCH_THRESHOLD,
        "plates_in_database": len(plates_db),
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "frame_skip": FRAME_SKIP,
        "total_time_s": round(total_time, 2),
        "fps_processing": round(
            processed_count / total_time, 2
        ) if total_time > 0 else 0,
        "unique_plates_count": len(unique_plates),
        "matched_plates_count": matched_count,
        "total_detections": len(all_detections),
        "unique_plates": unique_plates,
        "all_detections": all_detections
    }, f, indent=4, ensure_ascii=False)

# ==========================================================
# 📊 RÉSUMÉ FINAL
# ==========================================================
print("\n" + "=" * 60)
print("📊 RÉSUMÉ FINAL")
print("=" * 60)
print(f"  Vidéo             : {VIDEO_PATH}")
print(f"  Frames totales    : {total_frames}")
print(f"  Frames traitées   : {processed_count}")
print(f"  Temps total       : {total_time:.1f}s")
if total_time > 0:
    print(f"  Vitesse           : "
          f"{processed_count/total_time:.1f} fps")
print(f"  Détections totales: {len(all_detections)}")
print(f"  Plaques uniques   : {len(unique_plates)}")
print(f"  Plaques matchées  : {matched_count}/{len(unique_plates)}")
print(f"  Seuil matching    : {MATCH_THRESHOLD:.0%}")

# ── Détails par plaque ──
print("\n" + "-" * 60)
print("🏷  RÉSULTATS MATCHING :")
print("-" * 60)

for i, p in enumerate(unique_plates, 1):
    m = p["match"]
    emoji = get_match_emoji(m["status"])

    print(f"\n  #{i}  Détecté : {p['text']:15s}  "
          f"(OCR: {p['best_score']:.0%})")

    if m["matched"]:
        print(f"      {emoji} MATCH → {m['best_match']}  "
              f"(similarité: {m['score']:.1%})")
    else:
        print(f"      {emoji} {m['status']}  "
              f"(meilleur: {m['best_match']} "
              f"à {m['score']:.1%})")

    # Afficher top 3 scores
    if m.get("all_scores"):
        top = list(m["all_scores"].items())[:3]
        scores_str = ", ".join(
            f"{k}:{v:.0%}" for k, v in top
        )
        print(f"      Scores: {scores_str}")

# ── Légende ──
print("\n" + "-" * 60)
print("📋 LÉGENDE :")
print("-" * 60)
print(f"  ✅ EXACT     : similarité ≥ 95%")
print(f"  🟢 MATCH     : similarité ≥ {MATCH_THRESHOLD:.0%} (seuil)")
print(f"  🟡 PARTIAL   : similarité 50% - {MATCH_THRESHOLD:.0%}")
print(f"  🔴 NO_MATCH  : similarité < 50%")

# ── Fichiers ──
print("\n" + "-" * 60)
print("📁 FICHIERS :")
print("-" * 60)

if SAVE_VIDEO:
    print(f"  🎬 Vidéo  : {output_video}")
if SAVE_FRAMES:
    n_saved = len(os.listdir(frames_folder))
    print(f"  📸 Frames : {frames_folder}/ ({n_saved} images)")
if SAVE_CROPS:
    n_crops = len(os.listdir(crops_folder))
    print(f"  🔍 Crops  : {crops_folder}/ ({n_crops} images)")
print(f"  📄 JSON   : {json_path}")

print("\n✅ Terminé.")