import os
import torchvision.transforms as T

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR  = os.path.join(DATA_DIR, "Testing")

# ----- Training hyperparams -----
BATCH_SIZE = 32
EPOCHS = 15

# Phase 1 (head-only) and Phase 2 (fine-tune backbone)
LR_HEAD = 1e-3         # head warmup
LR_FINE_TUNE = 1e-4    # after unfreezing backbone
UNFREEZE_EPOCH = 3     # start fine-tuning after this epoch

# ----- Image settings (ResNet expects 224 + ImageNet stats) -----
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Augmentations for training; deterministic for test/val
transform_train = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_test = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
