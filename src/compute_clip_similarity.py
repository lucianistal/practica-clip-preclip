import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from utils import cosine_sim

if __name__ == "__main__":
    captions_file = "../dataset/captions.csv"
    images_root = "../dataset"   # here filenames in CSV are like "animales/bird.jpg"
    out_csv = "../results/clip_similarities.csv"

    df = pd.read_csv(captions_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_embeddings = []
    text_embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(images_root, row["filename"])
        image = Image.open(img_path).convert("RGB")
        # Procesamos texto e imagen por separado para evitar confusiones con batching mixto
        img_inputs = processor(images=image, return_tensors="pt").to(device)
        txt_inputs = processor(text=row["caption"], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            img_emb = model.get_image_features(**img_inputs)  # shape (1, D)
            txt_emb = model.get_text_features(**txt_inputs)   # shape (1, D)

        image_embeddings.append(img_emb.cpu())
        text_embeddings.append(txt_emb.cpu())

    img_embs = torch.cat(image_embeddings, dim=0)  # (N, D)
    txt_embs = torch.cat(text_embeddings, dim=0)   # (N, D)

    sim_matrix = cosine_sim(img_embs, txt_embs)  # (N, N)
    df_sim = pd.DataFrame(sim_matrix, columns=df["filename"], index=df["filename"])
    os.makedirs("../results", exist_ok=True)
    df_sim.to_csv(out_csv)
    print(f" Saved CLIP similarity matrix to {out_csv}")
