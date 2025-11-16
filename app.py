import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from src.models.resnet import build_model
from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

# Load model only once
@st.cache_resource
def load_model():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
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

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image and I’ll predict the tumor type using your trained ResNet model.")

uploaded_file = st.file_uploader("Upload MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    model = load_model()

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze()

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    st.subheader(f"🔍 Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("### Probability Breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")
