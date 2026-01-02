import pickle
import numpy as np

with open("face_db/face_db.pkl", "rb") as f:
    data = pickle.load(f)

print("Số embedding:", len(data["ids"]))
print("ID đầu tiên:", data["ids"][0])
print("Shape embedding:", data["embeddings"][0].shape)
print("Norm embedding:", np.linalg.norm(data["embeddings"][0]))
