import json
import csv
import re

with open("data/generated_captions.json", "r") as f:
    captions = json.load(f)

colors = [
    "black", "white", "red", "blue", "green", "yellow", "purple",
    "grey", "gray", "pink", "brown", "orange", "beige", "navy", "silver", "gold"
]

sleeve_map = {
    "long sleeves": "Full Sleeves",
    "long sleeved": "Full Sleeves",
    "long sleeve": "Full Sleeves",
    "short sleeve": "Half Sleeves",
    "half sleeve": "Half Sleeves",
    "sleeveless": "Sleeveless",
    "puff": "Puffed"
}

neckline_map = {
    "v - neck": "V-Neck",
    "round neck": "Round neck",
    "high neck": "High neck",
    "collar": "Collared",
    "collared": "Collared",
    "turtle": "Turtle Neck"
}

material_map = {
    "cotton": "Cotton",
    "satin": "Satin",
    "lace": "Lace"
}

brand_map = {
    "tommy": "Tommy Hilfiger",
    "adidas": "Adidas",
    "levi": "Levi's",
    "calvin": "Calvin Klein",
    "nike": "Nike",
    "adi": "Adidas"
}

type_map = {
    "shirt": "Shirt",
    "tee": "T-Shirt",
    "t - shirt": "T-Shirt",
    "polo": "Polo",
    "sweatshirt": "Sweatshirt",
    "sweater": "Sweater",
    "top": "Tops",
    "tank top": "Tank Top",
    "crop": "Cropped Top",
    "blouse": "Blouse",
    "bodysuit": "Bodysuit",
    "pregnant": "Maternity"
}

pattern_map = {
    "floral": "Floral",
    "leopard": "Leopard",
    "polka": "Polka",
    "plaid": "Plaid",
    "striped": "Striped",
    "stripes": "Striped",
    "graphic": "Graphic"
}

def normalize(text):
    return text.lower().strip()

def extract_metadata(caption):
    caption = normalize(caption)
    
    found_color = "Other"
    for color in colors:
        if color in caption:
            found_color = color.capitalize()
            break

    found_sleeve = "Unknown"
    for key in sleeve_map:
        if key in caption:
            found_sleeve = sleeve_map[key]
            break

    found_neckline = "Unknown"
    for key in neckline_map:
        if key in caption:
            found_neckline = neckline_map[key]
            break

    found_material = "Unknown"
    for key in material_map:
        if key in caption:
            found_material = material_map[key]
            break

    found_brand = "Unknown"
    for key in brand_map:
        if key in caption:
            found_brand = brand_map[key]
            break

    found_type = "Unknown"
    for key in type_map:
        if key in caption:
            found_type = type_map[key]
            break

    found_pattern = "Unknown"
    for key in pattern_map:
        if key in caption:
            found_pattern = pattern_map[key]
            break

    return found_color, found_sleeve, found_neckline, found_material, found_brand, found_type, found_pattern

with open("data/meta.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "color", "sleeve", "neckline", "material", "brand", "type", "pattern"])

    for img, caption in captions.items():
        color, sleeve, neckline, material, brand, type, pattern = extract_metadata(caption)
        writer.writerow([img, color, sleeve, neckline, material, brand, type, pattern])