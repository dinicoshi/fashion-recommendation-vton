# End-to-end Fashion Recommendation and Virtual Try-On System

# Introduction

This project is an end-to-end Fashion Recommendation and Virtual Try-On System built using deep learning and vision-language models. It supports multiple types of queries—text, image, and gallery-based selection—and recommends visually and semantically similar clothing items. It also allows users to try one various clothes on different models virtually.

# Project Structure

```
fashion_recommender/
├── app/
│   ├── app.py
├── autoencoder_model/
│   ├── cloth_recommender_resnet.pth   
│   ├── dataset.py           
│   ├── interface.py                
│   ├── model.py                 
│   └── train_autoencoder.py
├── clip_model/
│   ├── clip-finetuned/
│   ├── dataset.py          
│   ├── image_embeddings.npy     
│   ├── image_filenames.json           
│   ├── save_image_embeddings.py
│   ├── search.py
│   └── train_clip.py
├── data/
│   ├── test/
│   ├── train/   
│   ├── generate_caption.py
│   ├── generate_meta.py             
│   ├── meta.csv              
│   ├── test_pairs.txt               
│   └── train_pairs.txt          
├── features_additional/
│   ├── half_upcycle.py              
│   ├── overlap_upcycle.py 
│   └── patchwork_upcycle.py             
├── .gitignore         
├── requirements.txt          
└── README.md                 
```

# Models Used

The project implements various ML/DL models to recommend clothes based on images and text queries, as well as to try-on clothes virtually:

- Image-based Recommendation: custom ResNet50 (with custom Encoder-Decoder)
- Text-based Recommendation: BLIP (for image caption generation), CLIP (Fine Tuned - for quick text-to-image retrieval)
- User Interface: Streamlit
- Vrtual Try-On: CP-VTON

# UI Demo and Saved Models

The demo implementation/working of the project UI, as well as the saved models can be accessed using this link: https://drive.google.com/drive/folders/1mYER6ES-cVXdNTvlpnA79MhQ6Nt2XfUK?usp=drive_link  
Models: 
- cloth_recommender_resnet.pth
- model.safetensors