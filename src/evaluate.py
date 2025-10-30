import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__:
    preclip = pd.read_csv("../results/preclip_similarities.csv", index_col=0)
    clip = pd.read_csv("../results/clip_similarities.csv", index_col=0)

    diag_preclip = np.diag(preclip.values)
    diag_clip = np.diag(clip.values)

    print(" Mean similarity (image-caption pairs):")
    print(f"Pre-CLIP: {diag_preclip.mean():.3f}")
    print(f"CLIP:     {diag_clip.mean():.3f}")

    plt.figure()
    plt.bar(["Pre-CLIP", "CLIP"], [diag_preclip.mean(), diag_clip.mean()])
    plt.title("Average Image–Caption Similarity")
    plt.ylabel("Cosine similarity")
    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/similarity_comparison.png")
    print("Saved plot to ../results/similarity_comparison.png")

