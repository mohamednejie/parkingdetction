import cv2
import time
import os
import requests

# On importe le modèle et la fonction depuis notre fichier d'IA
from detect_plate import model_plate, read_license_plate

# ==========================================
# ⚙️ CONFIGURATION DE LA CAMÉRA
# ==========================================
CAMERA_STREAM = 0  # Remplacez 0 par "rtsp://..." pour votre vraie caméra IP
LARAVEL_URL = "http://127.0.0.1:8000/api/parking/entrance"
COOLDOWN_SECONDS = 10  # Temps d'attente entre 2 voitures

SAVE_FOLDER = "captures_parking"
os.makedirs(SAVE_FOLDER, exist_ok=True)

print("🎥 Connexion à la caméra...")
cap = cv2.VideoCapture(CAMERA_STREAM)

if not cap.isOpened():
    print("❌ Erreur : Impossible d'ouvrir la caméra.")
    exit()

last_capture_time = 0
print("✅ Caméra active. En attente d'une voiture...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Perte du flux vidéo. Reconnexion...")
        time.sleep(2)
        cap = cv2.VideoCapture(CAMERA_STREAM)
        continue

    current_time = time.time()

    # Si le temps de pause (cooldown) est passé, on cherche une plaque
    if current_time - last_capture_time > COOLDOWN_SECONDS:
        
        # YOLO cherche ultra-rapidement (sans OCR)
        results = model_plate.predict(source=frame, conf=0.40, device="cpu", verbose=False)
        
        if len(results[0].boxes) > 0:
            print("\n" + "="*40)
            print("📸 VÉHICULE DÉTECTÉ ! Capture en cours...")
            
            # 1. Prendre la photo et la sauvegarder
            photo_path = f"{SAVE_FOLDER}/capture_{int(current_time)}.jpg"
            cv2.imwrite(photo_path, frame)
            
            # 2. Lire la photo avec EasyOCR
            plate_text, confidence, msg = read_license_plate(photo_path)
            
            if plate_text:
                print(f"🔤 Plaque lue : {plate_text} (Score: {confidence})")
                
                # 3. Envoyer à Laravel
                try:
                    print("📡 Envoi à Laravel...")
                    response = requests.post(
                        LARAVEL_URL,
                        json={"plate_number": plate_text},
                        timeout=5
                    )
                    print(f"Réponse Laravel : {response.text}")
                except Exception as e:
                    print("❌ Erreur connexion Laravel :", e)
            else:
                print("❌ Plaque repérée mais le texte est illisible.")
            
            print(f"⏳ Pause de {COOLDOWN_SECONDS} secondes...")
            print("="*40)
            last_capture_time = time.time()

    # (Optionnel) Afficher l'image en direct
    cv2.imshow("Camera Entree Parking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Appuyez sur 'q' pour quitter
        break

cap.release()
cv2.destroyAllWindows()