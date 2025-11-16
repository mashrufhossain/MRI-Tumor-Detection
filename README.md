# 🧠 Brain Tumor MRI Classifier

A deep learning system for classifying brain MRI scans into **four categories**:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The model is a fine-tuned **ResNet34** implemented in PyTorch, trained on a combined multi-source dataset. A **Streamlit** interface provides fast, local inference.

---

## 🚀 Features

- **ResNet34** backbone with transfer learning
- Streamlit-based diagnostic UI
- Fully compatible with **CPU** and **MPS (Apple Silicon)**
- Clean, modular project structure

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your_repo_url>
cd dataset_kaggle
```

### 2. Activate virtual environment
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Structure

Project expects:

```
data/
  combined_training/
      glioma/
      meningioma/
      notumor/
      pituitary/

  combined_testing/
      glioma/
      meningioma/
      notumor/
      pituitary/
```

> These directories are `.gitignored` and must exist locally.

---

## 🧠 Training

Run training with:
```bash
python -m src.main
```

Model checkpoints are saved to:
```
weights/best_model.pth
```

---

## 📊 Evaluation

```bash
python -m src.eval.evaluate
```

Provides:
- Accuracy
- Classification report
- Confusion matrix

---

## 🩺 Streamlit App

Launch the local diagnostic UI:
```bash
streamlit run app.py
```

Allows:
- Uploading MRI images
- Viewing predictions and probabilities


## 🧱 Tech Stack

- Python 3.11
- PyTorch
- Torchvision
- Streamlit
- Pillow
- Matplotlib
- Kaggle API

---

## 📌 Future Work

- Grad-CAM visualization
- Experiment/version tracking
- Model comparison utilities
- Optional REST API backend

---

## ⭐ Summary

This repository contains a complete MRI tumor classification pipeline:
- Dataset preparation
- Model training with ResNet34
- Evaluation and reporting
- Streamlit-based inference UI
- Clean Python project organization

