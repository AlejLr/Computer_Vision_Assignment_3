import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import JesterDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.baseline_cnn import create_baseline_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Baseline CNN on Jester Dataset')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the frames folder for the Jester dataset')
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the train CSV file")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation CSV file")
    parser.add_argument('--labels_csv', type=str, required=True, help="Path to the labels CSV file")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default="experiments/baseline_model", help="Directory to save checkpoints and logs")
    parser.add_argument('--max_train_samples', type=int, default=None, help="Optional: limit number of training samples (for quick local tests).")
    parser.add_argument('--max_val_samples', type=int, default=None, help="Optional: limit number of validation samples (for quick local tests).")
    args = parser.parse_args()
    return args

def subset_dataset(dataset, max_samples):
    
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    dataset_subset = dataset
    dataset_subset.samples = dataset.samples[:max_samples]
    return dataset_subset

def create_data_loaders(data_root, train_csv, val_csv, labels_csv, batch_size, num_workers, max_train_samples=None, max_val_samples=None):
    
    train_set = JesterDataset(
        data_root=data_root,
        csv_path=train_csv,
        labels_csv_path=labels_csv,
        num_frames=8,
        temporal=True,
        transform=get_train_transforms(224),
        is_train=True,
    )
    val_set = JesterDataset(
        data_root=data_root,
        csv_path=val_csv,
        labels_csv_path=labels_csv,
        num_frames=8,
        temporal=True,
        transform=get_val_transforms(224),
        is_train=False,
    )
    
    # In case of quick tests
    #train_set = subset_dataset(train_set, max_train_samples)
    #val_set = subset_dataset(val_set, max_val_samples)
    
    num_classes = len(train_set.labels_names)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes

def train_one_epoch(model, dataLoader, criterion, optimizer, device, epoch):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataLoader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": running_loss / max(1, total),
            "acc": correct / max(1, total),
        })
    
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc

def evaluate(model, dataLoader, criterion, device, epoch):
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataLoader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                "loss": running_loss / max(1, total),
                "acc": correct / max(1, total),
            })
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = create_data_loaders(
        data_root=args.data_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    model = create_baseline_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, "model_best.pth")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch
        )
        
        print(
            f"Epoch {epoch}: "
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            best_model_path,
        )
        print(f"New best model saved at epoch {epoch} with val_acc: {val_acc:.4f}")
    
    print(f"Training completed. Best val_acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
