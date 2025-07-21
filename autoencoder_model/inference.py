import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import ClothDataset
from model import ResNetAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = ResNetAutoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_model/cloth_recommender_resnet.pth", map_location=device))
model.eval()

test_dataset = ClothDataset("data/test/cloth", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

features = []
paths = []

with torch.no_grad():
    for images, img_paths in test_loader:
        images = images.to(device)
        emb, _ = model(images)
        emb = emb.cpu().numpy().squeeze()
        features.append(emb)
        paths.append(img_paths[0])

features = np.vstack(features)

def recommend(query_path, top_k=10):
    image = Image.open(query_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb, _ = model(image)
        query_emb = query_emb.cpu().numpy().reshape(1, -1)

    dists = np.linalg.norm(features - query_emb, axis=1)
    top_idxs = np.argsort(dists)[:top_k]
    return [(paths[i], dists[i]) for i in top_idxs]

def show_results(query_path, result_paths_scores):
    fig, axes = plt.subplots(1, len(result_paths_scores)+1, figsize=(15, 5))
    query_img = Image.open(query_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title("Query")
    axes[0].axis('off')

    for i, (img_path, score) in enumerate(result_paths_scores):
        img = Image.open(img_path).convert('RGB')
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"Top {i+1}\n{score:.2f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

query_path = 'data/test/cloth/00401_00.jpg' 
results = recommend(query_path)
show_results(query_path, results)