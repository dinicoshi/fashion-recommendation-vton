import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("clip_model/clip-finetuned").to(device)
processor = CLIPProcessor.from_pretrained("clip_model/clip-finetuned")

image_dir = "data/train/cloth"
embeddings = []
filenames = []

for fname in tqdm(os.listdir(image_dir)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    image = Image.open(os.path.join(image_dir, fname)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
        filenames.append(fname)

embeddings = np.concatenate(embeddings, axis=0)
np.save("clip_model/image_embeddings.npy", embeddings)

with open("clip_model/image_filenames.json", "w") as f:
    json.dump(filenames, f)