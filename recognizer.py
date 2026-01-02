import faiss
import pickle
import numpy as np

class FaissFaceRecognizer:
    def __init__(self, db_path, threshold=0.55):
        with open(db_path, "rb") as f:
            data = pickle.load(f)

        self.ids = data["ids"]
        embeddings = np.vstack(data["embeddings"]).astype("float32")

        self.index = faiss.IndexFlatIP(512)
        self.index.add(embeddings)
        self.threshold = threshold

    def recognize(self, embedding):
        emb = embedding.astype("float32")[None, :]
        scores, idxs = self.index.search(emb, 1)

        score = float(scores[0][0])
        idx = int(idxs[0][0])

        if score >= self.threshold:
            return self.ids[idx], score
        return "unknown", score
