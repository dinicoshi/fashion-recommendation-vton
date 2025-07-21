import os
import json
import random
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from clip_model.search import model as clip_model, processor
from autoencoder_model.model import ResNetAutoencoder
from autoencoder_model.dataset import ClothDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "data/train/cloth"
TEST_IMAGE_DIR = "data/test/cloth"
EMBEDDINGS_PATH = "clip_model/image_embeddings.npy"
FILENAMES_PATH = "clip_model/image_filenames.json"
META_CSV_PATH = "data/meta.csv"

@st.cache_resource
def load_resources():
    try:
        ae = ResNetAutoencoder().to(DEVICE)
        ae.load_state_dict(torch.load("autoencoder_model/cloth_recommender_resnet.pth", map_location=DEVICE))
        ae.eval()

        img_embeddings = np.load(EMBEDDINGS_PATH)
        with open(FILENAMES_PATH, "r") as f:
            img_filenames = json.load(f)

        trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        df_meta = pd.read_csv(META_CSV_PATH)
        return ae, img_embeddings, img_filenames, trans, df_meta
    except FileNotFoundError as e:
        st.error(f"Error loading resource: {e}. Please ensure all model files and data are in place.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

autoencoder, image_embeddings, image_filenames, transform, meta_df = load_resources()

st.set_page_config(
    page_title="Fashion Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h2, h3, h4, h5, h6 {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stRadio > label > div {
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👗 Fashion Recommendation System 👕")

if "selected_gallery_image" not in st.session_state:
    st.session_state.selected_gallery_image = None
if "last_query_type" not in st.session_state:
    st.session_state.last_query_type = "Text Query"

st.markdown("---")
query_type = st.radio(
    "### Choose Your Recommendation Method:",
    ["Text Query", "Image Query", "Gallery Browse"],
    horizontal=True,
    help="Select how you want to find fashion recommendations."
)
st.markdown("---")

if query_type != st.session_state.last_query_type:
    st.session_state.selected_gallery_image = None
    st.session_state.last_query_type = query_type
    st.rerun()

top_k = st.slider("### Number of Recommendations to Show", min_value=1, max_value=10, value=5)

if query_type == "Text Query":
    st.header("✍️ Find Clothes by Text Description")
    st.info("Enter a description to discover matching fashion items.")
    query = st.text_input("Enter your fashion description here:", placeholder=" e.g. 'red sleeveless top', 'blue jeans jacket', 'beach top'")

    if st.button("Search for Text Query 🔎"):
        if not query:
            st.warning("❗ Please enter a valid text description to search.")
        else:
            with st.spinner("Searching for matches..."):
                inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    text_features = clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                image_embeddings_tensor = torch.tensor(image_embeddings).to(DEVICE)
                image_embeddings_tensor = image_embeddings_tensor / image_embeddings_tensor.norm(dim=-1, keepdim=True)

                similarities = torch.nn.functional.cosine_similarity(
                    text_features, image_embeddings_tensor
                )
                top_indices = similarities.topk(top_k).indices.cpu().numpy()

            st.subheader(f"✨ Top-{top_k} Matches for: '{query}'")
            if top_indices.size > 0:
                cols = st.columns(top_k)
                for i, idx in enumerate(top_indices):
                    fname = image_filenames[idx.item()]
                    score = similarities[idx].item()
                    img_path = os.path.join(IMAGE_DIR, fname)
                    if os.path.exists(img_path):
                        with Image.open(img_path) as img:
                            cols[i].image(img, caption=f"Similarity: {score:.2f}", use_container_width=True)
                    else:
                        cols[i].warning(f"Image not found: {fname}")
            else:
                st.info("No matches found for your query. Try a different description!")

elif query_type == "Image Query":
    st.header("🖼️ Find Similar Clothes by Uploading an Image")
    st.info("Upload an image of a clothing item, and we'll find similar styles from our collection.")
    uploaded_file = st.file_uploader("Upload your clothing image here:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Your Uploaded Image", width=250)

        with st.spinner("Finding similar items..."):
            query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_feat, _ = autoencoder(query_tensor)
                query_feat = query_feat.cpu().numpy().reshape(1, -1)

            dataset = ClothDataset(TEST_IMAGE_DIR, transform=transform)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            features, paths = [], []
            with torch.no_grad(): # Ensure this block encompasses the tensor operations
                for images, img_paths in loader:
                    images = images.to(DEVICE)
                    emb, _ = autoencoder(images)
                    # FIX: Add .detach() here to remove from computational graph
                    features.append(emb.cpu().detach().numpy().squeeze()) 
                    paths.append(img_paths[0])

            features = np.vstack(features)
            dists = np.linalg.norm(features - query_feat, axis=1)
            top_idxs = np.argsort(dists)[:top_k]

            result_paths = [paths[i] for i in top_idxs]
            result_scores = [dists[i] for i in top_idxs]

        st.subheader("🌟 Top Recommendations Based on Your Image:")
        if result_paths:
            cols = st.columns(top_k)
            for i in range(top_k):
                if os.path.exists(result_paths[i]):
                    with Image.open(result_paths[i]) as img:
                        cols[i].image(img, caption=f"Rank {i+1} | Distance: {result_scores[i]:.2f}", use_container_width=True)
                else:
                    cols[i].warning(f"Image not found: {os.path.basename(result_paths[i])}")
        else:
            st.info("No similar items found in the database. Try a different image!")

elif query_type == "Gallery Browse":
    st.header("🛍️ Browse Our Collection")
    st.info("Explore our diverse clothing gallery!")

    st.sidebar.markdown("### 🔍 Filter Your Search")

    color_filter = st.sidebar.multiselect("Color", meta_df["color"].dropna().unique(), help="Filter by dominant color.")
    sleeve_filter = st.sidebar.multiselect("Sleeve", meta_df["sleeve"].dropna().unique(), help="Filter by sleeve type.")
    neckline_filter = st.sidebar.multiselect("Neckline", meta_df["neckline"].dropna().unique(), help="Filter by neckline style.")
    material_filter = st.sidebar.multiselect("Material", meta_df["material"].dropna().unique(), help="Filter by fabric material.")
    brand_filter = st.sidebar.multiselect("Brand", meta_df["brand"].dropna().unique(), help="Filter by brand.")
    type_filter = st.sidebar.multiselect("Type", meta_df["type"].dropna().unique(), help="Filter by clothing type (e.g., dress, shirt).")
    pattern_filter = st.sidebar.multiselect("Pattern", meta_df["pattern"].dropna().unique(), help="Filter by pattern (e.g., floral, striped).")

    if st.sidebar.button("Reset Gallery Filters and Selection 🔄"):
        st.session_state.selected_gallery_image = None
        st.rerun()

    filtered_df = meta_df.copy()
    if color_filter: filtered_df = filtered_df[filtered_df["color"].isin(color_filter)]
    if sleeve_filter: filtered_df = filtered_df[filtered_df["sleeve"].isin(sleeve_filter)]
    if neckline_filter: filtered_df = filtered_df[filtered_df["neckline"].isin(neckline_filter)]
    if material_filter: filtered_df = filtered_df[filtered_df["material"].isin(material_filter)]
    if brand_filter: filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
    if type_filter: filtered_df = filtered_df[filtered_df["type"].isin(type_filter)]
    if pattern_filter: filtered_df = filtered_df[filtered_df["pattern"].isin(pattern_filter)]

    image_pool = filtered_df["image"].tolist()
    valid_gallery_images = [img for img in image_pool if os.path.exists(os.path.join(IMAGE_DIR, img))]
    
    if len(valid_gallery_images) == 0:
        st.warning("⚠️ No images found matching your selected filters. Please try adjusting them.")
        gallery_images_to_display = []
    else:
        current_selected_filename = None
        if st.session_state.selected_gallery_image:
            current_selected_filename = os.path.basename(st.session_state.selected_gallery_image)
            if current_selected_filename not in [os.path.basename(p) for p in valid_gallery_images]:
                st.session_state.selected_gallery_image = None

        gallery_images_to_display = []
        if st.session_state.selected_gallery_image:
            gallery_images_to_display.append(st.session_state.selected_gallery_image)
            remaining_valid_images = [os.path.join(IMAGE_DIR, img_name) for img_name in valid_gallery_images if os.path.basename(img_name) != current_selected_filename]
            num_to_sample = min(19, len(remaining_valid_images))
            gallery_images_to_display.extend(random.sample(remaining_valid_images, num_to_sample))
            random.shuffle(gallery_images_to_display)
        else:
            gallery_images_to_display = [os.path.join(IMAGE_DIR, img_name) for img_name in random.sample(valid_gallery_images, min(100, len(valid_gallery_images)))]

    st.subheader("🖼️ Our Curated Collection:")

    cols = st.columns(5)
    for i, img_path in enumerate(gallery_images_to_display):
        col_idx = i % 5
        try:
            with Image.open(img_path) as img:
                if cols[col_idx].button(
                    f"Select",
                    key=f"gallery_select_btn_{os.path.basename(img_path)}"
                ):
                    st.session_state.selected_gallery_image = img_path
                    st.rerun()

                cols[col_idx].image(img, use_container_width=True)
                
                caption_text = os.path.basename(img_path)
                if st.session_state.selected_gallery_image == img_path:
                    cols[col_idx].markdown(f"**Selected:** :green[{caption_text}]")
                else:
                    cols[col_idx].caption(caption_text)

        except FileNotFoundError:
            cols[col_idx].warning(f"Image not found: {os.path.basename(img_path)}")
        except Exception as e:
            cols[col_idx].error(f"Error loading {os.path.basename(img_path)}: {e}")

    st.markdown("---")

    if st.session_state.selected_gallery_image:
        st.subheader("✨ Top Recommendations Based on Your Selection:")
        
        selected_image_path = st.session_state.selected_gallery_image
        
        st.image(Image.open(selected_image_path).convert("RGB"), caption=f"Query Image: {os.path.basename(selected_image_path)}", width=250)

        with st.spinner("Generating recommendations..."):
            query_img_pil = Image.open(selected_image_path).convert("RGB")
            query_tensor = transform(query_img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                query_feat, _ = autoencoder(query_tensor)
                query_feat = query_feat.cpu().numpy().reshape(1, -1)
            
            query_feat_tensor = torch.tensor(query_feat).to(DEVICE)
            query_feat_tensor = query_feat_tensor / query_feat_tensor.norm(dim=-1, keepdim=True)

            image_embeddings_tensor_normalized = torch.tensor(image_embeddings).to(DEVICE)
            image_embeddings_tensor_normalized = image_embeddings_tensor_normalized / image_embeddings_tensor_normalized.norm(dim=-1, keepdim=True)

            similarities = torch.nn.functional.cosine_similarity(
                query_feat_tensor, image_embeddings_tensor_normalized
            )
            
            selected_img_filename_only = os.path.basename(selected_image_path)
            indexed_results = []
            for idx, (sim, fname) in enumerate(zip(similarities, image_filenames)):
                if fname != selected_img_filename_only:
                    indexed_results.append((sim.item(), os.path.join(IMAGE_DIR, fname)))

            indexed_results.sort(key=lambda x: x[0], reverse=True)
            top_k_results = indexed_results[:top_k]

            result_paths = [item[1] for item in top_k_results]
            result_scores = [item[0] for item in top_k_results]

        if result_paths:
            result_cols = st.columns(top_k)
            for i in range(top_k):
                if i < len(result_paths) and os.path.exists(result_paths[i]):
                    with Image.open(result_paths[i]) as img:
                        result_cols[i].image(img, caption=f"Rank {i+1} | Similarity: {result_scores[i]:.2f}", use_container_width=True)
                elif i < len(result_paths):
                    result_cols[i].warning(f"Image not found: {os.path.basename(result_paths[i])}")
                else:
                    result_cols[i].info("No more recommendations.")
        else:
            st.info("No recommendations found for the selected image in the current collection.")