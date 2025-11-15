import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        # classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # optionally freeze backbone (we’ll unfreeze later in training)
        self.set_backbone_requires_grad(not freeze_backbone)

    def set_backbone_requires_grad(self, requires_grad: bool):
        for name, param in self.backbone.named_parameters():
            # keep head trainable always
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = requires_grad

    def forward(self, x):
        return self.backbone(x)
