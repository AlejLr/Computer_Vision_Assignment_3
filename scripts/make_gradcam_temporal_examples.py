import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import JesterDataset
from src.data.transforms import get_val_transforms
from src.models.temporal_model import TemporalModel


def overlay_cam(image_rgb_uint8, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * image_rgb_uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "experiments/gradcam_temporal"
    os.makedirs(save_dir, exist_ok=True)

    num_frames = 8
    dataset = JesterDataset(
        data_root="data/raw/small-20bn-jester-v1",
        csv_path="data/raw/jester-v1-validation.csv",
        labels_csv_path="data/raw/jester-v1-labels.csv",
        num_frames=num_frames,
        temporal=True,
        transform=get_val_transforms(224),
        is_train=False,
    )

    model = TemporalModel(
        num_classes=len(dataset.labels_names),
        num_frames=num_frames,
        pretrained_backbone=True,
        hidden_size=512,
        num_layers=1,
    ).to(device)

    ckpt = torch.load("experiments/temporal_model/model_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    layer4 = model.backbone.features[-2]
    target_layer = layer4[-1].conv2

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        h2 = target_layer.register_full_backward_hook(bwd_hook)
    else:
        h2 = target_layer.register_backward_hook(bwd_hook)

    indices = [10, 200, 300]

    for idx in indices:
        clip, true_label = dataset[idx]
        x = clip.unsqueeze(0).to(device)
        x.requires_grad_(True)

        model.zero_grad()
        model.train()  
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred]
        score.backward()

        A = activations["value"].detach()
        G = gradients["value"].detach()

        overlays = []
        for t in range(num_frames):
            A_t = A[t]
            G_t = G[t]

            weights = G_t.mean(dim=(1, 2), keepdim=True)
            cam = (weights * A_t).sum(dim=0)
            cam = torch.relu(cam).cpu().numpy()

            cam = cv2.resize(cam, (clip.shape[3], clip.shape[2]))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            frame_rgb = np.array(to_pil_image(clip[t].cpu()))
            overlays.append(overlay_cam(frame_rgb, cam))

        true_name = dataset.labels_names[true_label]
        pred_name = dataset.labels_names[pred]

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"Temporal Grad-CAM (idx={idx}) | true={true_name} | pred={pred_name}", fontsize=12)

        for t, ax in enumerate(axes.flatten()):
            ax.imshow(overlays[t])
            ax.set_title(f"t={t}", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"temporal_gradcam_grid_{idx}_true-{true_name}_pred-{pred_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {out_path}")

    h1.remove()
    h2.remove()


if __name__ == "__main__":
    main()
