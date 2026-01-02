import os, cv2, time, numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from face_align import align_face
from arcface_embedding import ArcFaceEmbedding
from databasecode.recognizer import FaissFaceRecognizer

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.model = YOLO("../modelAI/yolov8n-face.pt")
        self.embedder = ArcFaceEmbedding(
            "../modelAI/arcface_ir_se50.onnx",
            num_threads=max(2, os.cpu_count()),
        )
        self.recognizer = FaissFaceRecognizer(
            "../face_db/face_db.pkl", threshold=0.55
        )

        self.detect_interval = 15
        self.input_size = 416
        self.running = True
        self.frame_id = 0

        self.last_results = []

        self.ui_fps = 0
        self.detect_ms = 0
        self._fps_count = 0
        self._fps_time = time.perf_counter()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_id += 1

            if self.frame_id % self.detect_interval == 0:
                self.last_results = self.detect(frame)

            self.draw(frame)
            self.update_fps()
            self.frame_ready.emit(frame)

        self.cap.release()

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))

        t0 = time.perf_counter()
        r = self.model(img, conf=0.5, device="cpu", verbose=False)[0]
        self.detect_ms = (time.perf_counter() - t0) * 1000

        if r.boxes is None:
            return []

        sx, sy = w / self.input_size, h / self.input_size
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()

        results = []
        for box, lm in zip(boxes, kpts):
            x1, y1, x2, y2 = (box * [sx, sy, sx, sy]).astype(int)

            lm = [(int(x*sx), int(y*sy)) for x, y in lm]
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            lm_local = [(x-x1, y-y1) for x, y in lm]
            aligned = align_face(face, lm_local)
            if aligned is None:
                continue

            emb = self.embedder.get_embedding(aligned)
            name, score = self.recognizer.recognize(emb)

            results.append((x1,y1,x2,y2,name,score,lm))
        return results

    def draw(self, frame):
        for x1,y1,x2,y2,name,score,lm in self.last_results:
            color = (0,255,0) if name!="unknown" else (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{name} {score:.2f}",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)
            for x,y in lm:
                cv2.circle(frame,(x,y),2,(255,0,0),-1)

        cv2.putText(frame,f"FPS: {self.ui_fps:.1f}",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(255,255,0),2)
        cv2.putText(frame,f"Detect: {self.detect_ms:.1f} ms",
                    (10,60),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,0),2)

    def update_fps(self):
        self._fps_count += 1
        now = time.perf_counter()
        if now - self._fps_time >= 1:
            self.ui_fps = self._fps_count / (now - self._fps_time)
            self._fps_count = 0
            self._fps_time = now

    def stop(self):
        self.running = False
        self.wait()


