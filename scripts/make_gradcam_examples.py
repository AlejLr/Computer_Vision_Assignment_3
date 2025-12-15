import os
import sys
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import JesterDataset
from src.data.transforms import get_val_transforms
from src.models.baseline_cnn import create_baseline_model
from src.utils.gradcam import GradCAM

def overlay_cam(image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * image)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "experiments/gradcam"
    os.makedirs(save_dir, exist_ok=True)

    dataset = JesterDataset(
        data_root="data/raw/small-20bn-jester-v1",
        csv_path="data/raw/jester-v1-validation.csv",
        labels_csv_path="data/raw/jester-v1-labels.csv",
        num_frames=1,
        temporal=False,
        transform=get_val_transforms(224),
        is_train=False,
    )

    model = create_baseline_model(num_classes=27, device=device)

    ckpt = torch.load("experiments/baseline_model/model_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = model.backbone.layer4[-1].conv2
    cam_generator = GradCAM(model, target_layer)

    indices = [10, 50, 100, 200, 300]

    for idx in indices:
        img, label = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        cam = cam_generator.generate(input_tensor)
        img_np = np.array(to_pil_image(img.cpu()))

        overlay  = overlay_cam(img_np, cam)

        out_path = os.path.join(
            save_dir,
            f"gradcam_{idx}_{dataset.labels_names[label]}.png",
        )
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()