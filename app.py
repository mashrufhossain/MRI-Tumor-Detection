import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from src.models.resnet import BrainTumorResNet
from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

# Select device (MPS for Apple Silicon, CPU fallback)
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Load model only once
@st.cache_resource
def load_model():
    model = BrainTumorResNet(num_classes=4, freeze_backbone=True)
    model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


# Preprocess transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


# ---------------------- UI ---------------------- #

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image and I’ll predict the tumor type using your trained ResNet model.")

uploaded_file = st.file_uploader("Upload MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Display resized image for faster UI
    preview = img.copy()
    preview.thumbnail((600, 600))
    st.image(preview, caption="Uploaded MRI")

    # Load model
    model = load_model()

    # ---------------------- Progress Bar ---------------------- #

    progress = st.progress(0)

    progress.progress(10)   # Starting…

    # Step 1 — Preprocess
    x = transform(img).unsqueeze(0).to(device)
    progress.progress(40)

    # Step 2 — Model forward pass
    with torch.no_grad():
        logits = model(x)
    progress.progress(70)

    # Step 3 — Softmax probabilities
    probs = F.softmax(logits, dim=1).squeeze()
    progress.progress(100)

    # ---------------------- Prediction Output ---------------------- #

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    # Output results
    st.subheader(f"🔍 Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("### Probability Breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")
