import torch
import pandas as pd
from utils import cosine_sim
import numpy as np
import os

if __name__ == "__main__":
    captions_file = "../dataset/captions.csv"
    img_emb_file = "../results/image_embeddings_preclip.pt"
    txt_emb_file = "../results/text_embeddings_preclip.pt"
    out_csv = "../results/preclip_similarities.csv"

    df = pd.read_csv(captions_file)
    img_embs_dict = torch.load(img_emb_file)   # dict filename -> tensor (1, D)
    txt_embs = torch.load(txt_emb_file)        # tensor (N, D)

    # Convertir a tensor ordenado según el CSV
    img_tensors = torch.stack([img_embs_dict[row["filename"]].squeeze(0) for _, row in df.iterrows()])
    txt_tensors = txt_embs  # (N, D)

    sim_matrix = cosine_sim(img_tensors, txt_tensors)  # (N, N)
    df_sim = pd.DataFrame(sim_matrix, columns=df["filename"], index=df["filename"])
    os.makedirs("../results", exist_ok=True)
    df_sim.to_csv(out_csv)
    print(f" Saved Pre-CLIP similarity matrix to {out_csv}")
