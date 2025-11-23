import torch.nn as nn
import torchvision.models as models

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512
        
    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return feat
    
class TemporalModel(nn.Module):
    def __init__(self, num_classes, num_frames = 8, pretrained_backbone=True, hidden_size=512 ,num_layers=1):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.gru = nn.GRU(
            input_size = self.backbone.out_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, but got {T}"
        
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        D = features.size(-1)
        features = features.view(B, T, D)
        
        gru_out, h_n = self.gru(features)
        last_hidden = h_n[-1]
        
        logits = self.classifier(last_hidden)
        return logits
    