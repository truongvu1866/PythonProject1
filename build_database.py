import os
import cv2
from ultralytics import YOLO

from face_align import align_face
from arcface_embedding import ArcFaceEmbedding
from face_database import FaceDatabase


# ===== PATH =====
FACE_MODEL = "yolov8n-face.pt"
ARC_MODEL = "arcface_ir_se50.onnx"
DATA_DIR = "../data"

# ===== LOAD MODELS =====
detector = YOLO(FACE_MODEL)
embedder = ArcFaceEmbedding(ARC_MODEL, num_threads=4)
face_db = FaceDatabase("../face_db/face_db.pkl")


def process_image(img_path, person_id):
    img = cv2.imread(img_path)
    if img is None:
        return

    results = detector(img, conf=0.5, verbose=False)

    if len(results[0].boxes) == 0:
        return

    # Lấy face đầu tiên
    kps = results[0].keypoints.xy[0].cpu().numpy()

    aligned = align_face(img, kps)

    embedding = embedder.get_embedding(aligned)

    face_db.add_face(person_id, embedding)
    print(f"[OK] {person_id} ← {os.path.basename(img_path)}")


def main():
    for person_id in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            process_image(img_path, person_id)


if __name__ == "__main__":
    main()
