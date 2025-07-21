import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("clip_model/clip-finetuned").to(device)
processor = CLIPProcessor.from_pretrained("clip_model/clip-finetuned")

image_embeddings = np.load("clip_model/image_embeddings.npy")
with open("clip_model/image_filenames.json", "r") as f:
    image_filenames = json.load(f)

query = "beach top"
inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

image_embeddings_tensor = torch.tensor(image_embeddings)
similarities = torch.nn.functional.cosine_similarity(
    text_features.cpu(), image_embeddings_tensor
)

top_k = 5
top_indices = similarities.topk(top_k).indices

for idx in top_indices:
    print(f"Match: {image_filenames[idx]} (Score: {similarities[idx].item():.4f})")

image_dir = "data/train/cloth"
plt.figure(figsize=(15, 5))
for i, idx in enumerate(top_indices):
    fname = image_filenames[idx]
    score = similarities[idx].item()
    img_path = os.path.join(image_dir, fname)
    image = Image.open(img_path).convert("RGB")

    plt.subplot(1, top_k, i + 1)
    plt.imshow(image)
    plt.title(f"{fname}\nScore: {score:.2f}")
    plt.axis("off")

plt.suptitle(f"Top-{top_k} Matches for: '{query}'", fontsize=16)
plt.tight_layout()
plt.show()