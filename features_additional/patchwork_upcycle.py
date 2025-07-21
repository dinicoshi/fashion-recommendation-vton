import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

DATA_DIR = "/kaggle/input/clothes/clothes_tryon_dataset"
CLOTH_DIR = str(Path(DATA_DIR) / "train" / "cloth")
MASK_DIR = str(Path(DATA_DIR) / "train" / "cloth-mask")
OUTPUT_DIR = "./upcycled_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_image(cloth_path, mask_path):
    cloth = cv2.imread(cloth_path)
    mask = cv2.imread(mask_path, 0)
    if cloth is None or mask is None:
        raise ValueError(f"Cannot load cloth or mask: {cloth_path}, {mask_path}")
    mask = np.where(mask > 128, 255, 0).astype(np.uint8)
    if cloth.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (cloth.shape[1], cloth.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cloth, mask

def get_masked_region(image_segment, mask_segment):
    mask_3_channel = cv2.cvtColor(mask_segment, cv2.COLOR_GRAY2BGR)
    masked_segment = cv2.bitwise_and(image_segment, mask_3_channel)
    return masked_segment

def create_patchwork_onto_base_design(cloth1_path, cloth2_path, mask1_path, mask2_path, output_path, num_patches=5):
    cloth1, mask1 = load_and_prepare_image(cloth1_path, mask1_path)
    cloth2, mask2 = load_and_prepare_image(cloth2_path, mask2_path)
    target_height, target_width = cloth2.shape[:2]
    cloth1 = cv2.resize(cloth1, (target_width, target_height))
    mask1 = cv2.resize(mask1, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    upcycled_garment = cloth2.copy()
    upcycled_mask = mask2.copy()
    patch_size = min(target_width, target_height) // 8

    for _ in range(num_patches):
        px = random.randint(0, target_width - patch_size)
        py = random.randint(0, target_height - patch_size)
        patch_from_cloth1 = cloth1[py:py+patch_size, px:px+patch_size]
        patch_mask_from_cloth1 = mask1[py:py+patch_size, px:px+patch_size]
        masked_patch = get_masked_region(patch_from_cloth1, patch_mask_from_cloth1)
        valid_y, valid_x = np.where(mask2[patch_size//2 : target_height - patch_size//2, 
                                          patch_size//2 : target_width - patch_size//2] == 255)
        if len(valid_y) == 0:
            continue
        rand_idx = random.randint(0, len(valid_y) - 1)
        target_center_y = valid_y[rand_idx] + patch_size // 2
        target_center_x = valid_x[rand_idx] + patch_size // 2
        target_py = max(0, target_center_y - patch_size // 2)
        target_px = max(0, target_center_x - patch_size // 2)
        target_py_end = min(target_height, target_py + patch_size)
        target_px_end = min(target_width, target_px + patch_size)
        current_patch_h = target_py_end - target_py
        current_patch_w = target_px_end - target_px
        if current_patch_h <= 0 or current_patch_w <= 0:
            continue
        resized_masked_patch = cv2.resize(masked_patch, (current_patch_w, current_patch_h), interpolation=cv2.INTER_AREA)
        resized_patch_mask = cv2.resize(patch_mask_from_cloth1, (current_patch_w, current_patch_h), interpolation=cv2.INTER_NEAREST)
        placement_roi_mask = mask2[target_py:target_py_end, target_px:target_px_end]
        final_patch_mask = cv2.bitwise_and(resized_patch_mask, placement_roi_mask)
        final_patch_mask_3_channel = cv2.cvtColor(final_patch_mask, cv2.COLOR_GRAY2BGR)
        roi_upcycled = upcycled_garment[target_py:target_py_end, target_px:target_px_end]
        upcycled_garment[target_py:target_py_end, target_px:target_px_end] = \
            np.where(final_patch_mask_3_channel == 255, resized_masked_patch, roi_upcycled)
        upcycled_mask[target_py:target_py_end, target_px:target_px_end] = \
            np.maximum(upcycled_mask[target_py:target_py_end, target_px:target_px_end], final_patch_mask)

    cv2.imwrite(output_path, upcycled_garment)
    cv2.imwrite(output_path.replace('.jpg', '_mask.jpg'), upcycled_mask)
    return upcycled_garment

def calculate_environmental_impact():
    textile_saved_per_garment = 0.2
    co2_saved_per_garment = 7.0
    water_saved_per_garment = 2000
    savings = {
        'textile_waste_kg': textile_saved_per_garment * 2,
        'co2_kg': co2_saved_per_garment,
        'water_liters': water_saved_per_garment
    }
    return savings

def visualize_results(cloth1_path, cloth2_path, combined_garment_path, impact):
    cloth1 = cv2.imread(cloth1_path)
    cloth2 = cv2.imread(cloth2_path)
    combined_garment = cv2.imread(combined_garment_path)
    if cloth1 is None or cloth2 is None or combined_garment is None:
        raise ValueError("Cannot load images for visualization")
    cloth1 = cv2.cvtColor(cloth1, cv2.COLOR_BGR2RGB)
    cloth2 = cv2.cvtColor(cloth2, cv2.COLOR_BGR2RGB)
    combined_garment = cv2.cvtColor(combined_garment, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cloth1)
    plt.title(f"Original 1: {Path(cloth1_path).name}", fontsize=10)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(cloth2)
    plt.title(f"Original 2: {Path(cloth2_path).name}", fontsize=10)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(combined_garment)
    plt.title(f"Upcycled Design\nTextile Saved: {impact['textile_waste_kg']:.1f} kg\nCO2 Saved: {impact['co2_kg']:.1f} kg\nWater Saved: {impact['water_liters']:.0f} L", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def upcycle_pipeline_two_garments():
    try:
        cloth_files = [f for f in os.listdir(CLOTH_DIR) if f.endswith('.jpg')]
        if len(cloth_files) < 2:
            raise ValueError(f"Need at least 2 .jpg files in {CLOTH_DIR}. Please check dataset path.")
        selected_files = random.sample(cloth_files, 2)
        cloth1_path = os.path.join(CLOTH_DIR, selected_files[0])
        cloth2_path = os.path.join(CLOTH_DIR, selected_files[1])
        mask1_path = os.path.join(MASK_DIR, selected_files[0])
        mask2_path = os.path.join(MASK_DIR, selected_files[1])
        output_path = os.path.join(OUTPUT_DIR, f"patchwork_{Path(cloth1_path).stem}_on_{Path(cloth2_path).stem}.jpg")
        print(f"Selected base garment: {selected_files[1]}")
        print(f"Selected garment for patches: {selected_files[0]}")
        combined_img = create_patchwork_onto_base_design(cloth1_path, cloth2_path, mask1_path, mask2_path, output_path, num_patches=7)
        print(f"Patchwork design saved at: {output_path}")
        impact = calculate_environmental_impact()
        print("\nEnvironmental Impact of Upcycling:")
        print(f"- Textile waste saved: {impact['textile_waste_kg']:.1f} kg")
        print(f"- CO2 emissions saved: {impact['co2_kg']:.1f} kg")
        print(f"- Water saved: {impact['water_liters']:.0f} liters")
        visualize_results(cloth1_path, cloth2_path, output_path, impact)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if not os.path.exists(CLOTH_DIR):
        print(f"Error: Dataset directory for clothes ({CLOTH_DIR}) does not exist. Please verify the path.")
    elif not os.path.exists(MASK_DIR):
        print(f"Error: Dataset directory for masks ({MASK_DIR}) does not exist. Please verify the path.")
    else:
        upcycle_pipeline_two_garments()