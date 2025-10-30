import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_NO_MKL"] = "1"

import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from tqdm import tqdm

torch.set_num_threads(1)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

captions_file = "../dataset/captions.csv"
df = pd.read_csv(captions_file)

texts = df["caption"].tolist()
inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embeddings = model.get_text_features(**inputs)  # (N, D)

os.makedirs("../results", exist_ok=True)
torch.save(text_embeddings, "../results/text_embeddings_clip.pt")
print(" Saved CLIP text embeddings to ../results/text_embeddings_clip.pt")
