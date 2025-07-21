import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = "/kaggle/input/clothes/clothes_tryon_dataset"
CLOTH_DIR = str(Path(DATA_DIR) / "train" / "cloth")
MASK_DIR = str(Path(DATA_DIR) / "train" / "cloth-mask")
OUTPUT_DIR = "./upcycled_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_color_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42) 
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def recommend_clothing(cloth_dir, base_cloth_path, max_images=50):
    base_color = extract_color_features(base_cloth_path)
    cloth_files = [f for f in os.listdir(cloth_dir) if f.endswith('.jpg')]

    cloth_files = [f for f in cloth_files if f != Path(base_cloth_path).name]
    if not cloth_files:
        raise ValueError("No other clothing items found in directory")

    cloth_files = cloth_files[:max_images]

    similarities = []
    for cloth in tqdm(cloth_files, desc="Processing clothes"):
        cloth_path = os.path.join(cloth_dir, cloth)
        try:
            color = extract_color_features(cloth_path)
            similarity = cosine_similarity([base_color], [color])[0][0]
            color_name = get_color_name(color)
            similarities.append((cloth, similarity, color_name))
        except ValueError as e:
            print(f"Skipping {cloth}: {e}")
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    if not similarities:
        raise ValueError("No valid clothing items for recommendation")
    return os.path.join(cloth_dir, similarities[0][0]), similarities[0][2]

def get_color_name(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    return "Other"

def create_upcycled_design(cloth1_path, cloth2_path, mask1_path, mask2_path, output_path):
    cloth1 = cv2.imread(cloth1_path)
    cloth2 = cv2.imread(cloth2_path)
    mask1 = cv2.imread(mask1_path, 0)
    mask2 = cv2.imread(mask2_path, 0)

    if cloth1 is None or cloth2 is None or mask1 is None or mask2 is None:
        raise ValueError(f"Cannot load images or masks: {cloth1_path}, {cloth2_path}, {mask1_path}, {mask2_path}")

    cloth2 = cv2.resize(cloth2, (cloth1.shape[1], cloth1.shape[0]))
    mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))

    body = cv2.bitwise_and(cloth1, cloth1, mask=mask1)
    sleeves = cv2.bitwise_and(cloth2, cloth2, mask=mask2)

    upcycled = cv2.addWeighted(body, 0.6, sleeves, 0.4, 0)

    upcycled_mask = cv2.bitwise_or(mask1, mask2)

    cv2.imwrite(output_path, upcycled)
    cv2.imwrite(output_path.replace('.jpg', '_mask.jpg'), upcycled_mask)
    return upcycled

def calculate_environmental_impact():
    fabric_weight = 0.2 
    co2_per_garment = 7.0  
    water_per_garment = 2000  

    savings = {
        'textile_waste_kg': fabric_weight * 2,
        'co2_kg': co2_per_garment,
        'water_liters': water_per_garment
    }
    return savings

def visualize_results(base_cloth_path, recommended_cloth_path, upcycled_path, impact):
    base_cloth = cv2.imread(base_cloth_path)
    recommended_cloth = cv2.imread(recommended_cloth_path)
    upcycled = cv2.imread(upcycled_path)

    if base_cloth is None or recommended_cloth is None or upcycled is None:
        raise ValueError("Cannot load images for visualization")

    base_cloth = cv2.cvtColor(base_cloth, cv2.COLOR_BGR2RGB)
    recommended_cloth = cv2.cvtColor(recommended_cloth, cv2.COLOR_BGR2RGB)
    upcycled = cv2.cvtColor(upcycled, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(base_cloth)
    plt.title(f"Base Cloth: {Path(base_cloth_path).name}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recommended_cloth)
    plt.title(f"Recommended Cloth: {Path(recommended_cloth_path).name}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(upcycled)
    plt.title(f"Upcycled Design\nTextile Saved: {impact['textile_waste_kg']} kg\nCO2 Saved: {impact['co2_kg']} kg\nWater Saved: {impact['water_liters']} L")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def upcycle_pipeline(base_cloth_path=None):
    try:
        if base_cloth_path is None:
            cloth_files = [f for f in os.listdir(CLOTH_DIR) if f.endswith('.jpg')]
            if not cloth_files:
                raise ValueError(f"No .jpg files found in {CLOTH_DIR}")
            base_cloth_path = os.path.join(CLOTH_DIR, cloth_files[0])

        if not os.path.exists(base_cloth_path):
            raise ValueError(f"Base cloth file does not exist: {base_cloth_path}")

        recommended_cloth, rec_color = recommend_clothing(CLOTH_DIR, base_cloth_path, max_images=50)
        print(f"Recommended: Combine {Path(base_cloth_path).name} with {Path(recommended_cloth).name} (Color: {rec_color})")

        mask1_path = str(Path(base_cloth_path).parent.parent / "cloth-mask" / Path(base_cloth_path).name)
        mask2_path = str(Path(recommended_cloth).parent.parent / "cloth-mask" / Path(recommended_cloth).name)
        output_path = os.path.join(OUTPUT_DIR, f"upcycled_{Path(base_cloth_path).stem}_{Path(recommended_cloth).stem}.jpg")

        upcycled_img = create_upcycled_design(
            base_cloth_path, recommended_cloth, mask1_path, mask2_path, output_path
        )
        print(f"Upcycled design saved at: {output_path}")

        impact = calculate_environmental_impact()
        print("Environmental Impact:")
        print(f"- Textile waste saved: {impact['textile_waste_kg']} kg")
        print(f"- CO2 emissions saved: {impact['co2_kg']} kg")
        print(f"- Water saved: {impact['water_liters']} liters")

        visualize_results(base_cloth_path, recommended_cloth, output_path, impact)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if not os.path.exists(CLOTH_DIR):
        print(f"Error: Dataset directory {CLOTH_DIR} does not exist. Please verify the path.")
    else:
        upcycle_pipeline()