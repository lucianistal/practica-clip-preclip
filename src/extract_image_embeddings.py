from PIL import Image
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_NO_MKL"] = "1"

from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
from tqdm import tqdm

torch.set_num_threads(1)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

captions_file = "../dataset/captions.csv"
images_root = "../dataset"

df = pd.read_csv(captions_file)
embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(images_root, row["filename"])
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embed = model.get_image_features(**inputs)  # shape (1, D)
    embeddings.append(embed)

embeddings = torch.cat(embeddings, dim=0)  # (N, D)
os.makedirs("../results", exist_ok=True)
torch.save(embeddings, "../results/image_embeddings_clip.pt")
print(" Saved CLIP image embeddings to ../results/image_embeddings_clip.pt")
