import torch
import torch.nn as nn
from torchvision import models

class BaselineResNet18(nn.Module):
    '''
    Baseline model using ResNet18 architecture, it looks at the center frame of each video.
    
    - For now, uses ImageNet pre-trained weights.
    - Replaces the final classification layer to match the number of gesture classes.
    - Intended for use with the JesterDataset which provides center frames.
    '''
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
def create_baseline_model(num_classes, device = None):
    model = BaselineResNet18(num_classes=num_classes, pretrained=True)
    if device:
        model.to(device)
    return model