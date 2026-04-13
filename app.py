# ============================================================================
# 🚀 SERVEUR AI PARKING - VERSION DIAGNOSTIC AMÉLIORÉ
# ✅ FIX : Affichage de la réponse brute Laravel avant parsing JSON
# ✅ FIX : Vérification du Content-Type
# ✅ FIX : Gestion des erreurs 500 HTML
# ============================================================================

import os
import time
import cv2
import requests
import numpy as np
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from datetime import datetime
from collections import defaultdict

from verif_park import is_parking
from detect_plate import read_license_plate, model_plate
from basic_parking import process_basic_parking_frame
from premium_parking import PremiumParkingTracker

# ============================================================================
# 🏗️ INITIALISATION
# ============================================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

premium_tracker = PremiumParkingTracker()

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================

LARAVEL_API_URL          = "http://127.0.0.1:8000/api"
LARAVEL_ENTRANCE_WEBHOOK = "http://127.0.0.1:8000/api/parking/entrance"
LARAVEL_EXIT_WEBHOOK     = "http://127.0.0.1:8000/api/parking/exit"
LARAVEL_ALERT_WEBHOOK    = "http://127.0.0.1:8000/api/parking/alert"

COOLDOWN_SECONDS = 10
SAVE_FOLDER      = "captures_parking"
os.makedirs(SAVE_FOLDER, exist_ok=True)

data_lock = threading.Lock()

ACTIVE_CAMERAS = defaultdict(dict)
PARKING_PLATES = defaultdict(list)
PARKING_CONFIG = defaultdict(lambda: {
    "mode":         "basic",
    "slots":        [],
    "is_premium":   False,
    "slots_loaded": False
})
PARKING_DATA = defaultdict(lambda: {
    "total_cars":        0,
    "free_spots":        0,
    "occupied_spots":    0,
    "infractions":       0,
    "last_plate":        None,
    "last_plate_time":   None,
    "last_plate_status": None,
    "last_update":       None,
    "recent_plates":     [],
    "detection_count":   0,
    "active_cameras":    []
})


# ============================================================================
# ✅ RÉCUPÉRER CONFIG DEPUIS LARAVEL
# ============================================================================

def fetch_parking_config_from_laravel(parking_id: int) -> dict:
    try:
        response = requests.get(
            f"{LARAVEL_API_URL}/parking/{parking_id}/config",
            timeout=5
        )
        if response.status_code == 200:
            config = response.json()
            print(f"✅ [Laravel] Config P{parking_id}: "
                  f"premium={config.get('is_premium')} | "
                  f"slots={config.get('slots_count', 0)}")
            return config
        else:
            print(f"❌ [Laravel] Erreur P{parking_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ [Laravel] Exception P{parking_id}: {e}")
        return None


# ============================================================================
# ✅ CONVERSION COORDONNÉES RELATIVES → PIXELS
# ============================================================================

def convert_slots_to_pixels(slots_relative: list, frame_width: int, frame_height: int) -> list:
    if not slots_relative:
        return []
    slots_pixels = []
    for polygon in slots_relative:
        polygon_px = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x_px = int(round(float(point[0]) * frame_width))
                y_px = int(round(float(point[1]) * frame_height))
                polygon_px.append([x_px, y_px])
        if len(polygon_px) >= 3:
            slots_pixels.append(polygon_px)
    return slots_pixels


# ============================================================================
# 🔧 HELPERS
# ============================================================================

def is_camera_active(parking_id: int, camera_type: str) -> bool:
    with data_lock:
        if parking_id not in ACTIVE_CAMERAS:
            return False
        if camera_type not in ACTIVE_CAMERAS[parking_id]:
            return False
        return ACTIVE_CAMERAS[parking_id][camera_type].get("active", False)


def get_gate_mode(parking_id: int) -> str:
    with data_lock:
        if parking_id in ACTIVE_CAMERAS and "gate" in ACTIVE_CAMERAS[parking_id]:
            return ACTIVE_CAMERAS[parking_id]["gate"].get("gate_mode", "entrance")
    return "entrance"


# ============================================================================
# ✅ UPDATE PLATE STATUS — helper centralisé
# ============================================================================

def update_plate_status(parking_id: int, plate_text: str, status: str, extra: dict = None):
    with data_lock:
        PARKING_DATA[parking_id]["last_plate_status"] = status

        if PARKING_PLATES[parking_id]:
            PARKING_PLATES[parking_id][0]["status"] = status
            if extra:
                for key, value in extra.items():
                    PARKING_PLATES[parking_id][0][key] = value

        PARKING_DATA[parking_id]["recent_plates"] = PARKING_PLATES[parking_id].copy()

    print(f"✅ [STATUS] P{parking_id} | {plate_text} → {status}" +
          (f" | {extra}" if extra else ""))


# ============================================================================
# 🎥 WORKER CAMÉRA GATE — VERSION AVEC DIAGNOSTIC AMÉLIORÉ
# ============================================================================

def gate_worker(parking_id: int, camera_url: str, camera_name: str):
    global ACTIVE_CAMERAS, PARKING_PLATES, PARKING_DATA

    cap               = None
    last_capture_time = 0
    display_text      = ""
    display_color     = (0, 255, 0)
    text_timer        = 0

    gate_mode = get_gate_mode(parking_id)

    gate_mode_label = "ENTREE" if gate_mode == "entrance" else "SORTIE"
    gate_mode_color = (0, 255, 0) if gate_mode == "entrance" else (0, 140, 255)

    print(f"🚀 [P{parking_id}][GATE][{gate_mode_label}] Worker démarré — {camera_name}")
    print(f"   📹 URL: {camera_url}")

    try:
        while is_camera_active(parking_id, "gate"):

            # ── Connexion caméra ──────────────────────────────────────────────
            if cap is None:
                print(f"🔗 [P{parking_id}][GATE] Connexion à {camera_url}...")
                source = int(camera_url) if str(camera_url).isdigit() else camera_url
                cap    = cv2.VideoCapture(source)

                if not cap.isOpened():
                    print(f"❌ [P{parking_id}][GATE] Échec connexion")
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "CONNECTION ERROR", (150, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(error_frame, f"P{parking_id} - {gate_mode_label}", (220, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                    with data_lock:
                        if parking_id in ACTIVE_CAMERAS and "gate" in ACTIVE_CAMERAS[parking_id]:
                            ACTIVE_CAMERAS[parking_id]["gate"]["frame"] = error_frame
                    cap = None
                    for _ in range(50):
                        if not is_camera_active(parking_id, "gate"):
                            break
                        time.sleep(0.1)
                    continue

                print(f"✅ [P{parking_id}][GATE] Connecté! Mode: {gate_mode_label}")
                time.sleep(1)

            # ── Lecture frame ─────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ [P{parking_id}][GATE] Signal perdu")
                cap.release()
                cap = None
                time.sleep(2)
                continue

            current_time  = time.time()
            display_frame = frame.copy()

            if display_text and current_time - text_timer > 5:
                display_text = ""

            # ── 🔥 DÉTECTION DE PLAQUE ────────────────────────────────────────
            if current_time - last_capture_time > COOLDOWN_SECONDS:
                try:
                    results = model_plate.predict(
                        source=frame, conf=0.20, device="cpu", verbose=False
                    )

                    if len(results[0].boxes) > 0:
                        print(f"\n{'='*60}")
                        print(f"📸 [P{parking_id}][GATE][{gate_mode_label}] VÉHICULE DÉTECTÉ!")

                        photo_path = f"{SAVE_FOLDER}/p{parking_id}_{gate_mode}_{int(time.time())}.jpg"
                        cv2.imwrite(photo_path, frame)

                        plate_text, confidence, msg = read_license_plate(photo_path)
                        print(f"   🧐 OCR: '{plate_text}' (conf: {confidence:.2f})")

                        if plate_text and len(plate_text) >= 5:
                            print(f"🔤 [P{parking_id}][GATE][{gate_mode_label}] Plaque: {plate_text}")

                            # ── Enregistrement local ──────────────────────────
                            with data_lock:
                                current_datetime = datetime.now()
                                plate_entry = {
                                    "plate":            plate_text,
                                    "time":             current_datetime.strftime("%H:%M:%S"),
                                    "date":             current_datetime.strftime("%Y-%m-%d"),
                                    "timestamp":        time.time(),
                                    "status":           "detected",
                                    "confidence":       confidence,
                                    "gate_mode":        gate_mode,
                                    "total_price":      None,
                                    "duration_minutes": None,
                                }
                                PARKING_PLATES[parking_id].insert(0, plate_entry)
                                if len(PARKING_PLATES[parking_id]) > 30:
                                    PARKING_PLATES[parking_id] = PARKING_PLATES[parking_id][:30]

                                PARKING_DATA[parking_id]["last_plate"]        = plate_text
                                PARKING_DATA[parking_id]["last_plate_time"]   = current_datetime.strftime("%H:%M:%S")
                                PARKING_DATA[parking_id]["last_plate_status"] = "detected"
                                PARKING_DATA[parking_id]["last_update"]       = time.time()
                                PARKING_DATA[parking_id]["recent_plates"]     = PARKING_PLATES[parking_id].copy()
                                PARKING_DATA[parking_id]["detection_count"]   += 1

                            # ── 📡 Webhook Laravel ────────────────────────────
                            webhook_url = (
                                LARAVEL_ENTRANCE_WEBHOOK
                                if gate_mode == "entrance"
                                else LARAVEL_EXIT_WEBHOOK
                            )

                            try:
                                print(f"📡 [Laravel][{gate_mode_label}] → {webhook_url}")
                                print(f"   📦 Payload: plate={plate_text}, parking_id={parking_id}")

                                laravel_response = requests.post(
                                    webhook_url,
                                    json={
                                        "plate_number": plate_text,
                                        "parking_id":   parking_id,
                                        "gate_mode":    gate_mode,
                                    },
                                    headers={
                                        'Content-Type': 'application/json',
                                        'Accept':       'application/json'
                                    },
                                    timeout=5
                                )

                                status_code = laravel_response.status_code
                                print(f"   📬 HTTP {status_code}")

                                # ✅ AFFICHER RÉPONSE BRUTE
                                response_text = laravel_response.text
                                print(f"   📄 Réponse brute (500 premiers chars):")
                                print(f"      {response_text[:500]}")
                                
                                content_type = laravel_response.headers.get('Content-Type', 'N/A')
                                print(f"   📋 Content-Type: {content_type}")

                                # ✅ VÉRIFIER SI C'EST DU JSON
                                if 'application/json' not in content_type:
                                    print(f"   ⚠️ Laravel a retourné {content_type} au lieu de JSON!")
                                    print(f"   🔴 Réponse complète:\n{response_text}")
                                    
                                    display_text  = f"[{plate_text}] ERREUR SERVEUR"
                                    display_color = (0, 0, 255)
                                    update_plate_status(
                                        parking_id, plate_text, 'server_error',
                                        extra={"http_status": status_code, "response": response_text[:200]}
                                    )
                                    text_timer = current_time
                                    last_capture_time = current_time
                                    continue

                                # ✅ PARSER LE JSON
                                try:
                                    data = laravel_response.json()
                                    print(f"   ✅ JSON parsé avec succès")
                                except ValueError as json_err:
                                    print(f"   ❌ ERREUR JSON : {json_err}")
                                    print(f"   🔴 Réponse complète:\n{response_text}")
                                    
                                    display_text  = f"[{plate_text}] REPONSE INVALIDE"
                                    display_color = (0, 0, 255)
                                    update_plate_status(
                                        parking_id, plate_text, 'invalid_response',
                                        extra={"error": str(json_err), "response": response_text[:200]}
                                    )
                                    text_timer = current_time
                                    last_capture_time = current_time
                                    continue

                                status       = data.get('status', 'unknown')
                                total_price  = data.get('total_price', 0)
                                duration_min = data.get('duration_minutes', 0)

                                print(f"   📊 Données: status='{status}' price={total_price} duration={duration_min}")

                                # ══════════════════════════════════════════════
                                # ✅ TRAITEMENT DES STATUTS
                                # ══════════════════════════════════════════════

                                if gate_mode == "entrance":
                                    if status == 'authorized':
                                        display_text  = f"[{plate_text}] ENTREE AUTORISEE"
                                        display_color = (0, 255, 0)
                                        update_plate_status(parking_id, plate_text, 'authorized')

                                    elif status == 'already_inside':
                                        display_text  = f"[{plate_text}] DEJA A L'INTERIEUR"
                                        display_color = (0, 165, 255)
                                        update_plate_status(parking_id, plate_text, 'already_inside')

                                    elif status == 'no_reservation':
                                        display_text  = f"[{plate_text}] PAS DE RESERVATION"
                                        display_color = (0, 0, 255)
                                        update_plate_status(parking_id, plate_text, 'no_reservation')

                                    elif status == 'unknown':
                                        display_text  = f"[{plate_text}] PLAQUE INCONNUE"
                                        display_color = (0, 0, 200)
                                        update_plate_status(parking_id, plate_text, 'unknown')

                                    else:
                                        display_text  = f"[{plate_text}] ACCES REFUSE"
                                        display_color = (0, 0, 255)
                                        update_plate_status(parking_id, plate_text, status)

                                else:  # exit
                                    if status == 'awaiting_payment':
                                        display_text  = f"[{plate_text}] PAYER: {total_price} TND ({duration_min}min)"
                                        display_color = (0, 0, 255)

                                        update_plate_status(
                                            parking_id, plate_text, 'awaiting_payment',
                                            extra={
                                                "total_price":      total_price,
                                                "duration_minutes": duration_min,
                                                "gate_mode":        gate_mode,
                                                "laravel_response": data,
                                            }
                                        )

                                    elif status == 'unknown':
                                        display_text  = f"[{plate_text}] INCONNU / PAS DE SEJOUR"
                                        display_color = (0, 0, 200)
                                        update_plate_status(parking_id, plate_text, 'unknown')

                                    else:
                                        print(f"   ⚠️ Statut sortie inattendu: '{status}'")
                                        display_text  = f"[{plate_text}] {status.upper()}"
                                        display_color = (255, 165, 0)
                                        update_plate_status(parking_id, plate_text, status)

                            except requests.exceptions.Timeout:
                                print(f"⏱️ [P{parking_id}] Laravel timeout (5s)")
                                display_text  = f"[{plate_text}] SERVEUR TIMEOUT"
                                display_color = (0, 165, 255)
                                update_plate_status(parking_id, plate_text, 'timeout')

                            except requests.exceptions.ConnectionError:
                                print(f"🔌 [P{parking_id}] Laravel injoignable")
                                display_text  = f"[{plate_text}] SERVEUR HORS LIGNE"
                                display_color = (0, 165, 255)
                                update_plate_status(parking_id, plate_text, 'offline')

                            except Exception as e:
                                print(f"❌ [P{parking_id}] Erreur webhook: {e}")
                                import traceback
                                traceback.print_exc()
                                display_text  = f"[{plate_text}] ERREUR WEBHOOK"
                                display_color = (255, 255, 255)
                                update_plate_status(parking_id, plate_text, 'error')

                            text_timer        = current_time
                            last_capture_time = current_time
                            print(f"{'='*60}\n")

                        else:
                            print(f"🗑️ Ignoré: '{plate_text}' (longueur < 5 chars)")
                            last_capture_time = current_time

                except Exception as e:
                    print(f"❌ [P{parking_id}][GATE] Erreur détection: {e}")
                    import traceback
                    traceback.print_exc()

            # ── Overlay vidéo ─────────────────────────────────────────────────
            if display_text:
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, 0), (w, 70), (0, 0, 0), -1)
                cv2.putText(display_frame, display_text, (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2, cv2.LINE_AA)

            cv2.putText(display_frame, datetime.now().strftime("%H:%M:%S"),
                        (display_frame.shape[1] - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"P{parking_id} [{gate_mode_label}]",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gate_mode_color, 1)
            cv2.circle(display_frame,
                       (display_frame.shape[1] - 15, 15), 6, gate_mode_color, -1)

            with data_lock:
                if parking_id in ACTIVE_CAMERAS and "gate" in ACTIVE_CAMERAS[parking_id]:
                    ACTIVE_CAMERAS[parking_id]["gate"]["frame"] = display_frame

            time.sleep(0.033)

    except Exception as e:
        print(f"❌ [P{parking_id}][GATE] Erreur critique: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        print(f"⏹️ [P{parking_id}][GATE][{gate_mode_label}] Worker terminé")


# ============================================================================
# 🎥 WORKER CAMÉRA ZONE (inchangé)
# ============================================================================

def zone_worker(parking_id: int, camera_url: str, camera_name: str):
    global ACTIVE_CAMERAS, PARKING_DATA, PARKING_CONFIG

    cap          = None
    slots_pixels = []
    slots_loaded = False

    print(f"🚀 [P{parking_id}][ZONE] Worker démarré — {camera_name}")

    try:
        while is_camera_active(parking_id, "zone"):

            if cap is None:
                print(f"🔗 [P{parking_id}][ZONE] Connexion à {camera_url}...")
                source = int(camera_url) if str(camera_url).isdigit() else camera_url
                cap    = cv2.VideoCapture(source)

                if not cap.isOpened():
                    print(f"❌ [P{parking_id}][ZONE] Échec connexion")
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "CONNECTION ERROR", (150, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    with data_lock:
                        if parking_id in ACTIVE_CAMERAS and "zone" in ACTIVE_CAMERAS[parking_id]:
                            ACTIVE_CAMERAS[parking_id]["zone"]["frame"] = error_frame
                    cap          = None
                    slots_loaded = False
                    for _ in range(50):
                        if not is_camera_active(parking_id, "zone"):
                            break
                        time.sleep(0.1)
                    continue

                print(f"✅ [P{parking_id}][ZONE] Connecté!")

                if not slots_loaded:
                    print(f"📡 [P{parking_id}][ZONE] Récupération config Laravel...")
                    laravel_config = fetch_parking_config_from_laravel(parking_id)

                    if laravel_config:
                        is_premium = laravel_config.get("is_premium", False)
                        has_slots  = laravel_config.get("has_slots", False)
                        slots_data = laravel_config.get("slots", [])

                        if is_premium and has_slots and len(slots_data) > 0:
                            mode = "premium"
                            print(f"✅ [P{parking_id}][ZONE] Mode PREMIUM — {len(slots_data)} slots")
                        else:
                            mode = "basic"
                            reason = "non premium" if not is_premium else "aucun slot" if not has_slots else "slots vides"
                            print(f"⚠️ [P{parking_id}][ZONE] Mode BASIC ({reason})")

                        with data_lock:
                            PARKING_CONFIG[parking_id]["mode"]         = mode
                            PARKING_CONFIG[parking_id]["slots"]        = slots_data
                            PARKING_CONFIG[parking_id]["is_premium"]   = is_premium
                            PARKING_CONFIG[parking_id]["slots_loaded"] = True
                    else:
                        print(f"⚠️ [P{parking_id}][ZONE] Laravel injoignable → BASIC forcé")
                        with data_lock:
                            PARKING_CONFIG[parking_id]["mode"]         = "basic"
                            PARKING_CONFIG[parking_id]["slots"]        = []
                            PARKING_CONFIG[parking_id]["is_premium"]   = False
                            PARKING_CONFIG[parking_id]["slots_loaded"] = True

                    slots_loaded = True

                time.sleep(1)

            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ [P{parking_id}][ZONE] Signal perdu")
                cap.release()
                cap          = None
                slots_loaded = False
                time.sleep(2)
                continue

            display_frame             = frame.copy()
            frame_height, frame_width = frame.shape[:2]

            try:
                with data_lock:
                    config = dict(PARKING_CONFIG[parking_id])

                mode           = config["mode"]
                slots_relative = config.get("slots", [])

                if mode == "premium" and len(slots_relative) > 0:
                    slots_pixels = convert_slots_to_pixels(slots_relative, frame_width, frame_height)
                    if len(slots_pixels) > 0:
                        display_frame, free, occ, ill = premium_tracker.process_frame(
                            frame.copy(), slots_pixels, LARAVEL_ALERT_WEBHOOK
                        )
                        with data_lock:
                            PARKING_DATA[parking_id]["free_spots"]     = free
                            PARKING_DATA[parking_id]["occupied_spots"] = occ
                            PARKING_DATA[parking_id]["infractions"]    = ill
                            PARKING_DATA[parking_id]["total_cars"]     = occ + ill
                else:
                    car_count, display_frame = process_basic_parking_frame(frame.copy())
                    with data_lock:
                        PARKING_DATA[parking_id]["total_cars"]     = car_count
                        PARKING_DATA[parking_id]["free_spots"]     = "N/A"
                        PARKING_DATA[parking_id]["occupied_spots"] = "N/A"
                        PARKING_DATA[parking_id]["infractions"]    = "N/A"

            except Exception as e:
                print(f"❌ [P{parking_id}][ZONE] Erreur: {e}")
                import traceback
                traceback.print_exc()

            mode_label = "PREMIUM" if mode == "premium" else "BASIC"
            mode_color = (0, 215, 255) if mode == "premium" else (180, 180, 180)

            cv2.putText(display_frame, datetime.now().strftime("%H:%M:%S"),
                        (display_frame.shape[1] - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"P{parking_id} ZONE [{mode_label}]",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)
            cv2.circle(display_frame, (display_frame.shape[1] - 15, 15), 6, (0, 255, 0), -1)

            with data_lock:
                if parking_id in ACTIVE_CAMERAS and "zone" in ACTIVE_CAMERAS[parking_id]:
                    ACTIVE_CAMERAS[parking_id]["zone"]["frame"] = display_frame
                    PARKING_DATA[parking_id]["last_update"]     = time.time()

            time.sleep(0.5)

    except Exception as e:
        print(f"❌ [P{parking_id}][ZONE] Erreur critique: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        print(f"⏹️ [P{parking_id}][ZONE] Worker terminé")


# ============================================================================
# 🌐 ROUTES API (inchangées)
# ============================================================================

@app.route("/api/health", methods=["GET"])
@cross_origin()
def health_check():
    with data_lock:
        parkings = {p_id: list(cams.keys()) for p_id, cams in ACTIVE_CAMERAS.items()}
    return jsonify({
        "status":          "healthy",
        "server":          "Flask AI Parking",
        "active_parkings": parkings,
        "timestamp":       time.time()
    }), 200


@app.route("/api/parking/<int:parking_id>/setup", methods=["POST", "OPTIONS"])
@cross_origin()
def api_parking_setup(parking_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    data = request.get_json()
    mode = data.get("mode", "basic")
    with data_lock:
        PARKING_CONFIG[parking_id]["mode"] = mode
    return jsonify({"success": True, "mode": mode}), 200


@app.route("/api/parking/<int:parking_id>/camera/start", methods=["POST", "OPTIONS"])
@cross_origin()
def api_start_camera(parking_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    data        = request.get_json()
    camera_type = data.get("camera_type")
    stream_url  = data.get("stream_url")
    camera_name = data.get("name", f"{camera_type} camera")
    gate_mode   = data.get("gate_mode", "entrance")

    if not camera_type or not stream_url:
        return jsonify({"success": False, "error": "camera_type et stream_url requis"}), 400
    if camera_type not in ["gate", "zone"]:
        return jsonify({"success": False, "error": "camera_type: gate ou zone"}), 400
    if camera_type == "gate" and gate_mode not in ["entrance", "exit"]:
        gate_mode = "entrance"

    with data_lock:
        if parking_id in ACTIVE_CAMERAS and camera_type in ACTIVE_CAMERAS[parking_id]:
            if ACTIVE_CAMERAS[parking_id][camera_type].get("active"):
                return jsonify({"success": False, "error": f"Caméra {camera_type} déjà active"}), 400

        ACTIVE_CAMERAS[parking_id][camera_type] = {
            "url":        stream_url,
            "name":       camera_name,
            "active":     True,
            "frame":      None,
            "started_at": time.time(),
            "gate_mode":  gate_mode,
        }
        PARKING_DATA[parking_id]["active_cameras"] = list(ACTIVE_CAMERAS[parking_id].keys())

    worker_func = gate_worker if camera_type == "gate" else zone_worker
    thread = threading.Thread(
        target=worker_func,
        args=(parking_id, stream_url, camera_name),
        daemon=True
    )
    thread.start()

    print(f"⚡ [P{parking_id}] {camera_type.upper()} [{gate_mode.upper()}] — {stream_url}")
    return jsonify({
        "success":     True,
        "parking_id":  parking_id,
        "camera_type": camera_type,
        "gate_mode":   gate_mode,
    }), 200


@app.route("/api/parking/<int:parking_id>/camera/stop", methods=["POST", "OPTIONS"])
@cross_origin()
def api_stop_camera(parking_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    data        = request.get_json()
    camera_type = data.get("camera_type")

    if not camera_type:
        return jsonify({"success": False, "error": "camera_type requis"}), 400

    with data_lock:
        if parking_id not in ACTIVE_CAMERAS:
            return jsonify({"success": False, "error": f"Parking {parking_id} non trouvé"}), 404
        if camera_type not in ACTIVE_CAMERAS[parking_id]:
            return jsonify({"success": False, "error": f"Caméra {camera_type} non trouvée"}), 404
        ACTIVE_CAMERAS[parking_id][camera_type]["active"] = False

    time.sleep(0.5)

    with data_lock:
        if parking_id in ACTIVE_CAMERAS and camera_type in ACTIVE_CAMERAS[parking_id]:
            del ACTIVE_CAMERAS[parking_id][camera_type]
        if parking_id in ACTIVE_CAMERAS and len(ACTIVE_CAMERAS[parking_id]) == 0:
            del ACTIVE_CAMERAS[parking_id]
            PARKING_DATA[parking_id]["active_cameras"] = []
        elif parking_id in ACTIVE_CAMERAS:
            PARKING_DATA[parking_id]["active_cameras"] = list(ACTIVE_CAMERAS[parking_id].keys())

    return jsonify({"success": True, "parking_id": parking_id, "camera_type": camera_type}), 200


@app.route("/api/parking/<int:parking_id>/cameras/status", methods=["GET"])
@cross_origin()
def api_cameras_status(parking_id):
    with data_lock:
        cameras = []
        if parking_id in ACTIVE_CAMERAS:
            for cam_type, cam_info in ACTIVE_CAMERAS[parking_id].items():
                cameras.append({
                    "type":      cam_type,
                    "name":      cam_info.get("name"),
                    "url":       cam_info.get("url"),
                    "active":    cam_info.get("active", False),
                    "gate_mode": cam_info.get("gate_mode"),
                    "has_frame": cam_info.get("frame") is not None
                })
    return jsonify({"success": True, "parking_id": parking_id, "cameras": cameras}), 200


@app.route("/api/parking/<int:parking_id>/live_status", methods=["GET"])
@cross_origin()
def api_live_status(parking_id):
    with data_lock:
        if parking_id not in ACTIVE_CAMERAS or len(ACTIVE_CAMERAS[parking_id]) == 0:
            return jsonify({
                "status":     "offline",
                "message":    "Aucune caméra active",
                "parking_id": parking_id,
                "timestamp":  time.time()
            }), 200

        data_copy = dict(PARKING_DATA[parking_id])
        data_copy["recent_plates"] = PARKING_PLATES[parking_id][:15]
        config = dict(PARKING_CONFIG[parking_id])

    return jsonify({
        "status":     "online",
        "mode":       config.get("mode", "basic"),
        "is_premium": config.get("is_premium", False),
        "data":       data_copy,
        "parking_id": parking_id,
        "timestamp":  time.time()
    }), 200


@app.route("/api/parking/<int:parking_id>/clear_history", methods=["POST"])
@cross_origin()
def api_clear_history(parking_id):
    with data_lock:
        PARKING_PLATES[parking_id] = []
        PARKING_DATA[parking_id]["recent_plates"]     = []
        PARKING_DATA[parking_id]["detection_count"]   = 0
        PARKING_DATA[parking_id]["last_plate"]        = None
        PARKING_DATA[parking_id]["last_plate_time"]   = None
        PARKING_DATA[parking_id]["last_plate_status"] = None
    return jsonify({"success": True}), 200


@app.route("/api/parking/<int:parking_id>/open_barrier", methods=["POST", "OPTIONS"])
@cross_origin()
def open_barrier(parking_id: int):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    data   = request.get_json() or {}
    plate  = data.get("plate", "")
    reason = data.get("reason", "unknown")

    print(f"🔓 [P{parking_id}] BARRIÈRE OUVERTE — Plaque: {plate} — Raison: {reason}")

    return jsonify({
        "success":    True,
        "message":    f"Barrière ouverte pour {plate}",
        "parking_id": parking_id,
        "plate":      plate,
        "reason":     reason,
    }), 200


def generate_camera_feed(parking_id: int, camera_type: str):
    while True:
        frame = None
        with data_lock:
            if parking_id in ACTIVE_CAMERAS and camera_type in ACTIVE_CAMERAS[parking_id]:
                frame = ACTIVE_CAMERAS[parking_id][camera_type].get("frame")

        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"⚠️ Stream error: {e}")
        else:
            waiting = np.zeros((480, 640, 3), dtype=np.uint8)
            label   = "ENTREE/SORTIE" if camera_type == "gate" else "ZONE PARKING"
            cv2.putText(waiting, f"P{parking_id} - {label}", (140, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(waiting, "Connexion...", (250, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            ret, buffer = cv2.imencode('.jpg', waiting)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.033)


@app.route("/api/parking/<int:parking_id>/camera/<camera_type>/stream")
@cross_origin()
def camera_stream(parking_id: int, camera_type: str):
    return Response(
        generate_camera_feed(parking_id, camera_type),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma':        'no-cache',
            'Expires':       '0'
        }
    )


@app.route("/api/parking/<int:parking_id>/reset", methods=["POST", "OPTIONS"])
@cross_origin()
def reset_parking(parking_id):
    if request.method == "OPTIONS":
        return "", 200

    with data_lock:
        if parking_id in ACTIVE_CAMERAS:
            for cam_type in list(ACTIVE_CAMERAS[parking_id].keys()):
                ACTIVE_CAMERAS[parking_id][cam_type]["active"] = False
            time.sleep(0.3)
            del ACTIVE_CAMERAS[parking_id]

        PARKING_DATA[parking_id] = {
            "total_cars": 0, "free_spots": 0, "occupied_spots": 0, "infractions": 0,
            "last_plate": None, "last_plate_time": None, "last_plate_status": None,
            "last_update": None, "recent_plates": [], "detection_count": 0, "active_cameras": []
        }
        PARKING_PLATES[parking_id] = []
        PARKING_CONFIG[parking_id] = {
            "mode": "basic", "slots": [], "is_premium": False, "slots_loaded": False
        }

    return jsonify({"success": True, "message": f"Parking {parking_id} reset"}), 200


@app.route("/api/reset_all", methods=["POST", "OPTIONS"])
@cross_origin()
def reset_all():
    if request.method == "OPTIONS":
        return "", 200

    with data_lock:
        for parking_id in list(ACTIVE_CAMERAS.keys()):
            for cam_type in list(ACTIVE_CAMERAS[parking_id].keys()):
                ACTIVE_CAMERAS[parking_id][cam_type]["active"] = False
        time.sleep(0.3)
        ACTIVE_CAMERAS.clear()
        PARKING_DATA.clear()
        PARKING_PLATES.clear()
        PARKING_CONFIG.clear()

    return jsonify({"success": True, "message": "All reset"}), 200


@app.route("/api/debug/parking/<int:parking_id>", methods=["GET"])
@cross_origin()
def debug_parking(parking_id):
    with data_lock:
        data   = dict(PARKING_DATA.get(parking_id, {}))
        plates = PARKING_PLATES.get(parking_id, [])[:5]
        config = dict(PARKING_CONFIG.get(parking_id, {}))
        cams   = {}
        if parking_id in ACTIVE_CAMERAS:
            for t, info in ACTIVE_CAMERAS[parking_id].items():
                cams[t] = {
                    "active":    info.get("active"),
                    "gate_mode": info.get("gate_mode"),
                    "has_frame": info.get("frame") is not None,
                }
    return jsonify({
        "parking_id":    parking_id,
        "parking_data":  data,
        "recent_plates": plates,
        "config":        config,
        "cameras":       cams,
    }), 200


# ============================================================================
# 🚀 DÉMARRAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SERVEUR IA PARKING — VERSION DIAGNOSTIC AMÉLIORÉ")
    print("="*70)
    print("✅ Affichage de la réponse brute Laravel (500 chars)")
    print("✅ Vérification du Content-Type avant parsing JSON")
    print("✅ Gestion des erreurs 500 HTML de Laravel")
    print()
    print("🔍 En cas d'erreur, vous verrez maintenant :")
    print("   - La réponse complète de Laravel")
    print("   - Le Content-Type reçu")
    print("   - Le code HTTP exact")
    print("="*70 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)