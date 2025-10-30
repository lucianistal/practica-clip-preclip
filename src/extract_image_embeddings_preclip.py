# Extrae embeddings de imagen usando una ResNet pretrained (sin cabeza)
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga ResNet50 y elimina la cabeza (fc)
resnet = models.resnet18(pretrained=True)  # resnet18 suficiente y más ligera
modules = list(resnet.children())[:-1]  # quitar la capa fc final
backbone = torch.nn.Sequential(*modules).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

captions_file = "../dataset/captions.csv"
images_root = "../dataset"
df = pd.read_csv(captions_file)

embeddings = {}

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(images_root, row["filename"])
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        feat = backbone(x)  # shape (1, 512, 1, 1) for resnet18
        feat = feat.reshape(feat.size(0), -1).cpu()  # (1, D)
        embeddings[row["filename"]] = feat  # keep as tensor of shape (1, D)

os.makedirs("../results", exist_ok=True)
torch.save(embeddings, "../results/image_embeddings_preclip.pt")
print("Saved pre-CLIP image embeddings to ../results/image_embeddings_preclip.pt")
