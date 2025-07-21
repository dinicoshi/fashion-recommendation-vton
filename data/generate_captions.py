from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

image_dir = "data/train/cloth"
captions = {}

for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image = Image.open(os.path.join(image_dir, fname)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions[fname] = caption
    print(f"{fname} -> {caption}")

with open("data/generated_captions.json", "w") as f:
    json.dump(captions, f, indent=2)