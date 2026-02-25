import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

def is_parking(image_path):
    # ======================
    # 0️⃣ CHARGEMENT DU MODÈLE
    # ======================
    model = YOLO("model/yolo26n.pt")
    results = model(image_path, verbose=False)

    # ======================
    # 1️⃣ DÉTECTION VÉHICULES
    # ======================
    car_count = 0
    for r in results:
        if r.boxes is not None:
            for cls in r.boxes.cls.cpu().numpy():
                if int(cls) in [2, 3, 5, 7]:  # voiture, moto, bus, camion
                    car_count += 1

    # ======================
    # 2️⃣ LECTURE IMAGE
    # ======================
    img = cv2.imread(image_path)
    if img is None:
        return False, 0, {}

    h, w = img.shape[:2]
    if h < 400 or w < 400:
        return False, 0, {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ======================
    # 3️⃣ DÉTECTION DES LIGNES
    # ======================
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=110,
        minLineLength=120,
        maxLineGap=30
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if length > 100 and angle < 12:
                horizontal_lines.append(l)

    line_count = len(horizontal_lines)

    # ======================
    # 4️⃣ ANALYSE STRUCTURE PARKING
    # ======================
    grid_score = 0
    spacing_std = None

    if line_count >= 10:
        grid_score += 1

    if line_count >= 6:
        y_coords = sorted([l[0][1] for l in horizontal_lines])
        spacings = np.diff(y_coords)
        if len(spacings) >= 3:
            spacing_std = np.std(spacings)
            if spacing_std < 35:
                grid_score += 1

    if line_count > 0:
        y_min = min(l[0][1] for l in horizontal_lines)
        y_max = max(l[0][1] for l in horizontal_lines)
        coverage = (y_max - y_min) / h
        if coverage > 0.35:
            grid_score += 1

    regular_structure = grid_score >= 2

    # ======================
    # 5️⃣ CONTEXTE COULEUR
    # ======================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    asphalt_mask = cv2.inRange(hsv, (0, 0, 0), (180, 70, 130))
    asphalt_ratio = np.sum(asphalt_mask > 0) / (h * w)

    sand_mask = cv2.inRange(hsv, (18, 40, 60), (40, 255, 230))
    sand_ratio = np.sum(sand_mask > 0) / (h * w)

    # ======================
    # 6️⃣ SCORE FINAL
    # ======================
    score = 0
    diagnostics = []

    # Structure parking
    if line_count >= 15:
        score += 25
        diagnostics.append(f"{line_count} lignes")
    elif line_count >= 8:
        score += 15
        diagnostics.append(f"{line_count} lignes")

    if regular_structure:
        score += 20
        diagnostics.append("🅿️ structure parking")
    else:
        diagnostics.append("structure partielle")

    # Asphalte
    if asphalt_ratio > 0.25:
        score += 15
        diagnostics.append("asphalte")

    # Voitures
    if car_count >= 5:
        score += 30
        diagnostics.append(f"{car_count} voitures")
    elif car_count >= 2:
        score += 15
        diagnostics.append(f"{car_count} voiture(s)")
    else:
        diagnostics.append(f"{car_count} voiture(s) → trop peu pour confirmer")

    # Pénalités sable
    if sand_ratio > 0.25:
        score -= 40
        diagnostics.append("❌ sable")

    # ======================
    # 7️⃣ DÉCISION FINALE
    # ======================
    print("🔍", " | ".join(diagnostics))
    if spacing_std is not None:
        print(f"📐 grid std: {spacing_std:.1f}")

    if car_count < 2:
        is_parking_flag = False
    else:
        is_parking_flag = score >= 70

    # ======================
    # 8️⃣ ENREGISTREMENT JSON
    # ======================
    result = {
        "image": os.path.basename(image_path),
        "is_parking": is_parking_flag,
        "score": score,
        "car_count": car_count,
        "line_count": line_count,
        "regular_structure": regular_structure,
        "asphalt_ratio": float(asphalt_ratio),
        "sand_ratio": float(sand_ratio),
        "spacing_std": float(spacing_std) if spacing_std is not None else None,
        "diagnostics": diagnostics
    }

    json_path = "resultverifparlk/verifparking.json"
    # Si le fichier existe, charger les anciens résultats
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return is_parking_flag, score, result


# ======================
# 🧪 TEST
# ======================
flag, score, result = is_parking("data/test.jpg")

if flag:
    print(f"✅ PARKING (score: {score}/100)")
else:
    print(f"❌ NON-PARKING (score: {score}/100)")
