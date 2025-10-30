# Usa sentence-transformers para obtener embeddings de texto (multilingual suggested)
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import torch

model_name = "distiluse-base-multilingual-cased"
model = SentenceTransformer(model_name)

captions_file = "../dataset/captions.csv"
df = pd.read_csv(captions_file)
texts = df["caption"].tolist()

embs = model.encode(texts, convert_to_tensor=True)  # shape (N, D)
os.makedirs("../results", exist_ok=True)
torch.save(embs, "../results/text_embeddings_preclip.pt")
print("Saved pre-CLIP text embeddings to ../results/text_embeddings_preclip.pt")
