import os
import pickle
import numpy as np


class FaceDatabase:
    def __init__(self, db_path="face_db/face_db.pkl"):
        self.db_path = db_path
        self.embeddings = []
        self.ids = []

        if os.path.exists(db_path):
            self.load()

    def load(self):
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.ids = data["ids"]

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "ids": self.ids
            }, f)

    def add_face(self, face_id, embedding):
        self.embeddings.append(embedding.astype(np.float32))
        self.ids.append(face_id)
        self.save()
