import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

model = load_model()

# Load features and paths
@st.cache_data
def load_data():
    features = np.load('features.npy')
    paths = np.load('paths.npy', allow_pickle=True)
    return features, paths

features, paths = load_data()

st.title("🔍 CBIR: Content-Based Image Retrieval")
st.write("Upload an image to find visually similar images from the Corel dataset.")

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Query Image', use_column_width=True)

    # Extract feature
    img = img.resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    query_feat = model.predict(x).flatten()

    # Similarity
    sims = cosine_similarity([query_feat], features)[0]
    top_indices = np.argsort(sims)[::-1][:5]

    st.subheader("Top 5 Similar Images:")
    cols = st.columns(5)
    for i, idx in enumerate(top_indices):
        sim_score = sims[idx]
        result_img = Image.open(paths[idx])
        cols[i].image(result_img.resize((224,224)), caption=f"{sim_score:.2f}", use_column_width=True)
