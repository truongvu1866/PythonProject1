import cv2
import numpy as np
import time
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self,
                 model_path="yolov8n-face.pt",
                 detect_interval=5,
                 input_size=640):
        super().__init__()

        # ===== Camera =====
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ===== YOLO =====
        self.model = YOLO(model_path)

        # ===== Control =====
        self.running = True
        self.detect_interval = detect_interval
        self.input_size = input_size
        self.frame_id = 0

        # ===== Cache detect result =====
        self.last_boxes = []
        self.last_landmarks = []

        # ===== FPS & timing =====
        self.ui_fps = 0.0
        self.yolo_fps = 0.0
        self.last_detect_ms = 0.0

        self._fps_frame_count = 0
        self._fps_last_time = time.perf_counter()

    # ================= MAIN THREAD LOOP =================
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_id += 1
            h, w = frame.shape[:2]

            # ---- Detect mỗi N frame ----
            if self.frame_id % self.detect_interval == 0:
                small = cv2.resize(frame, (self.input_size, self.input_size))
                self.last_boxes, self.last_landmarks = self.detect(
                    small, w, h
                )

            # ---- Draw result + overlay ----
            self.draw(frame, self.last_boxes, self.last_landmarks)
            self.update_ui_fps()

            self.frame_ready.emit(frame)

        self.cap.release()

    # ================= YOLO DETECT =================
    def detect(self, img, orig_w, orig_h):
        t0 = time.perf_counter()

        results = self.model(
            img,
            conf=0.5,
            iou=0.4,
            device="cpu",
            verbose=False
        )

        t1 = time.perf_counter()

        # ---- Detect timing ----
        self.last_detect_ms = (t1 - t0) * 1000.0
        if self.last_detect_ms > 0:
            self.yolo_fps = 1000.0 / self.last_detect_ms

        boxes_out = []
        landmarks_out = []

        r = results[0]
        if r.boxes is None:
            return boxes_out, landmarks_out

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        boxes = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy() if r.keypoints else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            boxes_out.append((
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ))

            if kps is not None:
                lm = []
                for x, y in kps[i]:
                    lm.append((
                        int(x * scale_x),
                        int(y * scale_y)
                    ))
                landmarks_out.append(lm)

        return boxes_out, landmarks_out

    # ================= DRAW =================
    def draw(self, frame, boxes, landmarks):
        # ---- Draw boxes & landmarks ----
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            if i < len(landmarks):
                for x, y in landmarks[i]:
                    cv2.circle(frame, (x, y),
                               3, (0, 0, 255), -1)

        # ---- Overlay text ----
        cv2.putText(frame,
                    f"UI FPS: {self.ui_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Detect: {self.last_detect_ms:.1f} ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.putText(frame,
                    f"YOLO FPS: {self.yolo_fps:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    # ================= UI FPS =================
    def update_ui_fps(self):
        self._fps_frame_count += 1
        now = time.perf_counter()

        if now - self._fps_last_time >= 1.0:
            self.ui_fps = self._fps_frame_count / (now - self._fps_last_time)
            self._fps_frame_count = 0
            self._fps_last_time = now

    # ================= STOP =================
    def stop(self):
        self.running = False
        self.wait()
