import cv2
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = Path("./model/yolo26s.pt")
VIDEO_PATH = Path("./data/parking1.mp4")
OUTPUT_DIR = Path("./resultats_parking")
TARGET_CLASSES = [2]  # Voitures

# ==========================================
# VÉRIFICATIONS
# ==========================================
if not MODEL_PATH.exists():
    print(f"❌ Modèle non trouvé : {MODEL_PATH}")
    sys.exit(1)

if not VIDEO_PATH.exists():
    print(f"❌ Vidéo non trouvée : {VIDEO_PATH}")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CHARGER MODÈLE
# ==========================================
print("🚀 Chargement du modèle...")
model = YOLO(MODEL_PATH)
print("✅ Modèle chargé")

# ==========================================
# OUVRIR VIDÉO
# ==========================================
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir : {VIDEO_PATH}")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"\n📹 Vidéo : {VIDEO_PATH.name}")
print(f"   Résolution : {w}x{h}")
print(f"   FPS : {fps}")
print(f"   Durée : {duration:.1f}s ({total_frames} frames)")

# ==========================================
# ★ ENREGISTREMENT VIDÉO RÉSULTAT
# ==========================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_video = OUTPUT_DIR / f"resultat_{VIDEO_PATH.stem}_{timestamp}.avi"

video_writer = cv2.VideoWriter(
    str(output_video),
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

if video_writer.isOpened():
    print(f"   🔴 Enregistrement → {output_video.name}")
else:
    print("   ⚠️ Erreur création vidéo de sortie")

print(f"\n   Q = Quitter | S = Screenshot | P = Pause\n")


# ==========================================
# FONCTIONS
# ==========================================
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
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30)
    return enhanced


def enhance_twilight(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_conf(mode):
    if mode == 'NUIT':
        return 0.15
    elif mode == 'CREPUSCULE':
        return 0.20
    return 0.25


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def draw_frame(frame, boxes, mode, brightness, car_count,
               fps_val, max_count, current_frame, total_f, vid_fps):
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
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ─── Bandeau haut ───
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # ─── REC indicator ───
    if int(time.time() * 3) % 2 == 0:
        cv2.circle(frame, (20, 20), 6, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (32, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Temps vidéo
    current_time = current_frame / vid_fps if vid_fps > 0 else 0
    total_time = total_f / vid_fps if vid_fps > 0 else 0
    cv2.putText(frame, f"{format_time(current_time)} / {format_time(total_time)}",
               (75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ─── Mode jour/nuit ───
    mode_config = {
        'JOUR':       {"color": (0, 255, 255)},
        'CREPUSCULE': {"color": (0, 180, 255)},
        'NUIT':       {"color": (255, 200, 100)}
    }
    cfg = mode_config.get(mode, mode_config['JOUR'])

    mode_x = 270
    cv2.circle(frame, (mode_x, 20), 12, cfg['color'], -1)
    cv2.circle(frame, (mode_x, 20), 12, (255, 255, 255), 1)
    if mode == 'NUIT':
        cv2.circle(frame, (mode_x + 4, 17), 8, (0, 0, 0), -1)

    cv2.putText(frame, mode, (mode_x + 18, 26),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg['color'], 2)

    # ─── Compteur principal ───
    cv2.putText(frame, f"VOITURES : {car_count}",
               (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ─── Infos ───
    info = f"Lum: {brightness:.0f}/255 | FPS: {fps_val:.0f} | Max: {max_count}"
    cv2.putText(frame, info,
               (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Contrôles
    cv2.putText(frame, "Q:Quitter | P:Pause | S:Screenshot",
               (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # ─── Barre luminosité ───
    bar_x = fw - 180
    bar_w = 160
    bar_h = 18
    cv2.rectangle(frame, (bar_x, 10), (bar_x + bar_w, 10 + bar_h), (50, 50, 50), -1)
    fill = int(bar_w * (brightness / 255))
    cv2.rectangle(frame, (bar_x, 10), (bar_x + fill, 10 + bar_h), cfg['color'], -1)
    cv2.rectangle(frame, (bar_x, 10), (bar_x + bar_w, 10 + bar_h), (255, 255, 255), 1)

    # ─── Compteur coin droit ───
    box_x = fw - 110
    box_y = 40
    cv2.rectangle(frame, (box_x, box_y), (box_x + 90, box_y + 55), (0, 80, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + 90, box_y + 55), (0, 255, 0), 2)
    cv2.putText(frame, f"{car_count}", (box_x + 15, box_y + 42),
               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    # ─── Barre de progression vidéo (bas) ───
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, fh - 35), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    # Barre progression
    prog_x = 10
    prog_w = fw - 20
    prog_y = fh - 20
    prog_h = 8

    cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + prog_h),
                 (80, 80, 80), -1)

    if total_f > 0:
        progress = current_frame / total_f
        fill_prog = int(prog_w * progress)
        cv2.rectangle(frame, (prog_x, prog_y), (prog_x + fill_prog, prog_y + prog_h),
                     (0, 200, 255), -1)
        # Curseur
        cv2.circle(frame, (prog_x + fill_prog, prog_y + prog_h // 2), 6,
                  (255, 255, 255), -1)

    cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + prog_h),
                 (255, 255, 255), 1)

    # Pourcentage
    pct = (current_frame / total_f * 100) if total_f > 0 else 0
    cv2.putText(frame, f"Frame {current_frame}/{total_f} ({pct:.0f}%)",
               (10, fh - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return frame


# ==========================================
# ★ BOUCLE VIDÉO
# ==========================================
frame_count = 0
prev_time = time.time()
max_count = 0
stats = []
last_mode = 'JOUR'
last_brightness = 128
last_count = 0
last_boxes = None
skip = 1
paused = False
start_time = time.time()

print("🚀 Analyse en cours...\n")

while cap.isOpened():
    # ─── Pause ───
    if paused:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('p'):
            paused = False
            print("   ▶️ Reprise")
        elif key == ord('q'):
            break
        continue

    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"\n🏁 Fin de la vidéo ({frame_count} frames)")
        break

    frame_count += 1

    # ★ Sauter des frames pour CPU
    if frame_count % (skip + 1) != 0:
        # Dessiner avec les dernières données
        display = draw_frame(
            frame, last_boxes, last_mode, last_brightness,
            last_count, 0, max_count,
            frame_count, total_frames, fps
        )
        video_writer.write(display)
        cv2.imshow("Parking - Video + REC", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ─── Détecter jour/nuit ───
    if frame_count % 30 == 1:
        mode, brightness = detect_condition(frame)
        last_mode = mode
        last_brightness = brightness
    else:
        mode = last_mode
        brightness = last_brightness

    # ─── Amélioration nuit ───
    if mode == 'NUIT':
        detect_frame = enhance_night(frame)
    elif mode == 'CREPUSCULE':
        detect_frame = enhance_twilight(frame)
    else:
        detect_frame = frame

    # ─── Détection YOLO ───
    conf = get_conf(mode)
    results = model.predict(
        source=detect_frame,
        classes=TARGET_CLASSES,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    boxes = results[0].boxes
    car_count = len(boxes) if boxes is not None else 0
    last_count = car_count
    last_boxes = boxes

    if car_count > max_count:
        max_count = car_count

    # FPS
    now = time.time()
    fps_val = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
    prev_time = now

    # ─── Dessiner ───
    frame = draw_frame(
        frame, boxes, mode, brightness,
        car_count, fps_val, max_count,
        frame_count, total_frames, fps
    )

    # ★ ENREGISTRER
    video_writer.write(frame)

    # ─── Afficher ───
    cv2.imshow("Parking - Video + REC", frame)

    # ─── Log ───
    if frame_count % 50 == 0:
        progress = frame_count / total_frames * 100 if total_frames > 0 else 0
        print(f"   {progress:5.1f}% | Frame {frame_count}/{total_frames} "
              f"| {mode} | Voitures: {car_count} | Max: {max_count}")

    # Stats
    stats.append({
        "frame": frame_count,
        "mode": mode,
        "voitures": car_count,
        "luminosite": round(brightness, 1)
    })

    # ─── Contrôles ───
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n⏹️ Arrêt")
        break
    elif key == ord('p'):
        paused = True
        print("   ⏸️ Pause")
    elif key == ord('s'):
        path = OUTPUT_DIR / f"screenshot_{mode}_{frame_count}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"   📸 {path.name}")


# ==========================================
# FIN
# ==========================================
cap.release()
video_writer.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time

# ==========================================
# RÉSULTATS
# ==========================================
if stats:
    counts = [s['voitures'] for s in stats]
    modes_list = [s['mode'] for s in stats]

    jour_cars = [s['voitures'] for s in stats if s['mode'] == 'JOUR']
    nuit_cars = [s['voitures'] for s in stats if s['mode'] == 'NUIT']
    crep_cars = [s['voitures'] for s in stats if s['mode'] == 'CREPUSCULE']

    # Taille fichier vidéo
    video_size = 0
    if output_video.exists():
        video_size = output_video.stat().st_size / (1024 * 1024)

    result_data = {
        "video_source": str(VIDEO_PATH),
        "video_resultat": str(output_video),
        "taille_mb": round(video_size, 1),
        "frames_totales": len(stats),
        "temps_traitement": round(total_time, 1),
        "max_voitures": max(counts),
        "min_voitures": min(counts),
        "moyenne": round(np.mean(counts), 1),
        "par_mode": {
            "JOUR": {
                "frames": modes_list.count('JOUR'),
                "moyenne": round(np.mean(jour_cars), 1) if jour_cars else 0
            },
            "NUIT": {
                "frames": modes_list.count('NUIT'),
                "moyenne": round(np.mean(nuit_cars), 1) if nuit_cars else 0
            },
            "CREPUSCULE": {
                "frames": modes_list.count('CREPUSCULE'),
                "moyenne": round(np.mean(crep_cars), 1) if crep_cars else 0
            }
        },
        "timestamp": datetime.now().isoformat()
    }

    json_path = OUTPUT_DIR / "stats_video.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 55}")
    print(f"📊 RÉSULTAT FINAL")
    print(f"{'=' * 55}")
    print(f"   📹 Source        : {VIDEO_PATH.name}")
    print(f"   📹 Résultat      : {output_video.name} ({video_size:.1f} MB)")
    print(f"   ⏱️ Traitement    : {format_time(total_time)}")
    print(f"   🎞️ Frames        : {len(stats)}")
    print(f"")
    print(f"   🚗 Max voitures  : {result_data['max_voitures']}")
    print(f"   🚗 Min voitures  : {result_data['min_voitures']}")
    print(f"   🚗 Moyenne       : {result_data['moyenne']}")
    print(f"")
    print(f"   ☀️ JOUR          : {modes_list.count('JOUR')} frames"
          f" | Moy: {round(np.mean(jour_cars), 1) if jour_cars else 0}")
    print(f"   🌆 CREPUSCULE    : {modes_list.count('CREPUSCULE')} frames"
          f" | Moy: {round(np.mean(crep_cars), 1) if crep_cars else 0}")
    print(f"   🌙 NUIT          : {modes_list.count('NUIT')} frames"
          f" | Moy: {round(np.mean(nuit_cars), 1) if nuit_cars else 0}")
    print(f"")
    print(f"   📄 Stats JSON    : {json_path}")
    print(f"{'=' * 55}")