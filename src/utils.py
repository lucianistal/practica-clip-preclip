import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def load_image(path, size=224):
    """Carga una imagen y la convierte a tensor normalizado (para ResNet o CLIP)."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # shape (1, C, H, W)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """
    Calcula la similitud coseno entre dos conjuntos de embeddings.
    Devuelve un numpy.ndarray de shape (len(a), len(b)).
    Acepta tensores en CPU o GPU; realiza las operaciones en CPU al final.
    """
    # Asegurar tensores
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")

    # mover a CPU si es necesario (hacemos cálculo en CPU para estabilidad)
    a = a.detach().cpu()
    b = b.detach().cpu()

    # Normalizar (evitar división por cero)
    a_norm = a / (a.norm(dim=1, keepdim=True).clamp(min=1e-8))
    b_norm = b / (b.norm(dim=1, keepdim=True).clamp(min=1e-8))

    sim = (a_norm @ b_norm.T).numpy()  # shape (N, M)
    return sim