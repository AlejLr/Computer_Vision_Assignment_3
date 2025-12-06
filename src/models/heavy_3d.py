import torch.nn as nn

try:
    from torchvision.models.video import (
        r3d_18,
        R3D_18_Weights,
        r2plus1d_18,
        R2Plus1D_18_Weights,
    )
    HAS_NEW_VIDEO_API = True
except ImportError:
    from torchvision.models.video import r3d_18, r2plus1d_18
    R3D_18_Weights = None
    R2Plus1D_18_Weights = None
    HAS_NEW_VIDEO_API = False

class Heavy3DModel(nn.Module):
    def __init__(self, num_classes, backbone = "r3d_18", pretrained=True):

        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "r3d_18":
            if HAS_NEW_VIDEO_API and pretrained:
                weights = R3D_18_Weights.KINETICS400_V1
                model = r3d_18(weights=weights)
            else:
                model = r3d_18(pretrained=pretrained)

        elif self.backbone_name == "r2plus1d_18":
            if HAS_NEW_VIDEO_API and pretrained:
                weights = R2Plus1D_18_Weights.KINETICS400_V1
                model = r2plus1d_18(weights=weights)
            else:
                model = r2plus1d_18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'r3d_18' or 'r2plus1d_18'.")
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)
    
def create_heavy_3d_model(num_classes, device, backbone="r3d_18", pretrained=True):
    model = Heavy3DModel(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    return model.to(device)