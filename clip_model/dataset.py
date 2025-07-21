import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, processor):
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.image_files = list(self.captions.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        caption = self.captions[image_file]
        image = Image.open(os.path.join(self.image_dir, image_file)).convert("RGB")
        return {
            "image": image,
            "text": caption
        }