import torch
from torch import nn


class MedicalCNN(nn.Module):
    def __init__(self, architecture="resnet18", pretrained=True):
        super().__init__()
        from torchvision import models

        self.architecture = architecture
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            dimension = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.layer_name = "layer3"
        elif architecture == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            dimension = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.layer_name = "features.denseblock3"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.weights_name = (
            "IMAGENET1K_V1" if pretrained else "random initialization (test only)"
        )
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(dimension, 1)
        self.eval()

    @property
    def target_layer(self):
        return self.backbone.get_submodule(self.layer_name)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MedicalCNN(checkpoint["config"]["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
