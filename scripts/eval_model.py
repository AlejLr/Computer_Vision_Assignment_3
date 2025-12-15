import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import JesterDataset
from src.data.transforms import get_val_transforms
from src.models.baseline_cnn import create_baseline_model
from src.models.temporal_model import TemporalModel

def parse_args():
    p = argparse.ArgumentParser("Evaluate baseline or temporal model on Jester validation set")
    p.add_argument("--data_root", type=str, required=True, help="Root dir containing per-video frame folders")
    p.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    p.add_argument("--labels_csv", type=str, required=True, help="Path to labels CSV")
    p.add_argument("--model_type", type=str, required=True, choices=["baseline", "temporal"])
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to model_best.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=8, help="Used only for temporal model")
    p.add_argument("--save_dir", type=str, default="experiments/eval", help="Where to save evaluation outputs")
    return p.parse_args()

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    model.eval()

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total = 0
    correct = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        for t, p in zip(y.view(-1),preds.view(-1)):
            conf[t.long(), p.long()] += 1

        pbar.set_postfix({"Acc": f"{correct/total:.4f}"})

    accuracy = correct / max(total, 1)
    return accuracy, conf.cpu().numpy()

def per_class_accuracy(confusion_matrix):
    
    correct = np.diag(confusion_matrix)
    support = confusion_matrix.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(support > 0, correct / support, 0.0)
    return acc, support

def save_confusion_matrix(confusion_matrix, class_names, save_path, normalize=True):
    
    cm = confusion_matrix.astype(np.float32)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm,np.maximum(row_sums, 1.0))

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=6)
    plt.yticks(ticks, class_names, fontsize=6)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    is_temporal = args.model_type == "temporal"
    val_set = JesterDataset(
        data_root=args.data_root,
        csv_path=args.val_csv,
        labels_csv_path=args.labels_csv,
        num_frames=(args.num_frames if is_temporal else 1),
        temporal=is_temporal,
        transform=get_val_transforms(224),
        is_train=False,
    )
    class_names = val_set.labels_names
    num_classes = len(class_names)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.model_type == "baseline":
        model = create_baseline_model(num_classes=num_classes, device=device)
    else:
        model = TemporalModel(
            num_classes=num_classes,
            num_frames=args.num_frames,
            pretrained_backbone=True,
            hidden_size=512,
            num_layers=1,
        ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    acc, conf_mat = evaluate_model(model, val_loader, device, num_classes)
    pc_acc, support = per_class_accuracy(conf_mat)

    print(f"\nOverall validation accuracy: {acc:.4f}")

    metrics_path = os.path.join(args.save_dir, f"metrics_{args.model_type}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"model_type: {args.model_type}\n")
        f.write(f"ckpt_path: {args.ckpt_path}\n")
        f.write(f"overall_val_acc: {acc:.6f}\n")

    csv_path = os.path.join(args.save_dir, f"per_class_accuracy_{args.model_type}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_index", "class_name", "support", "accuracy"])
        for i, name in enumerate(class_names):
            w.writerow([i, name, int(support[i]), float(pc_acc[i])])

    cm_path = os.path.join(args.save_dir, f"confusion_matrix_{args.model_type}.png")
    save_confusion_matrix(conf_mat, class_names, cm_path, normalize=True)

    order = np.argsort(pc_acc)
    print("\nWorst 5 classes by per-class accuracy:")
    for i in order[:5]:
        print(f"  {i:02d} {class_names[i]:30s} acc={pc_acc[i]:.3f}  support={support[i]}")

    print("\nBest 5 classes by per-class accuracy:")
    for i in order[-5:][::-1]:
        print(f"  {i:02d} {class_names[i]:30s} acc={pc_acc[i]:.3f}  support={support[i]}")

    print(f"\nSaved:\n  {metrics_path}\n  {csv_path}\n  {cm_path}\n")

if __name__ == '__main__':
    main()