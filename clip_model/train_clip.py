import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from tqdm import tqdm
from dataset import ImageCaptionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

dataset = ImageCaptionDataset("data/train/cloth", "data/generated_captions.json", processor)

def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    return processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-6)

model.train()
for epoch in range(5):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(dataloader):.4f}")

model.save_pretrained("clip_model/clip-finetuned")
processor.save_pretrained("clip_model/clip-finetuned")