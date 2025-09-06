# model.py
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes: int):
    model = models.resnet18(weights=None)   # simple backbone
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model