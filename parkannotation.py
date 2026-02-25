import cv2
from ultralytics import solutions

# Video capture
cap = cv2.VideoCapture("data/parking1.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))

print(f"📹 Vidéo originale : {w}x{h} @ {fps} FPS")

# ★ CORRECTION 1 : Définir la taille cible UNE SEULE FOIS
# cv2.resize prend (largeur, hauteur) → (width, height)
TARGET_W = 1250
TARGET_H = 600

video_writer = cv2.VideoWriter(
    "parking management.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (TARGET_W, TARGET_H)  # ← Doit correspondre au resize
)

# Initialize parking management object
parkingmanager = solutions.ParkingManagement(
    model="./model/yolo26n.pt",
    json_file="bounding_boxes.json",
)

frame_count = 0

while cap.isOpened():
    ret, im0 = cap.read()

    # ★ CORRECTION 2 : Vérifier ret AVANT de toucher à im0
    if not ret or im0 is None:
        print(f"🏁 Fin de la vidéo après {frame_count} frames")
        break

    # ★ CORRECTION 3 : Resize APRÈS la vérification
    im0 = cv2.resize(im0, (TARGET_W, TARGET_H))

    results = parkingmanager(im0)

    video_writer.write(results.plot_im)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"   ⏳ Frame {frame_count} traitée...")

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"✅ Terminé ! {frame_count} frames traitées")