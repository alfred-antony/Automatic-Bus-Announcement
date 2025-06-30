import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from dataset import YOLODataset
from torchvision import transforms as T
import os
from pathlib import Path


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid_indices = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
        if not valid_indices:
            continue

        images = [images[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f'Epoch: {epoch} | Batch: {batch_idx}/{len(data_loader)} | Loss: {avg_loss:.4f} | Valid: {len(valid_indices)}/{len(targets)}')

    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    # Dataset setup
    train_root = "C:/Users/LEGION/Pictures/datasets/train"
    val_root = "C:/Users/LEGION/Pictures/datasets/val"

    train_dataset = YOLODataset(train_root, get_transform(train=True))
    val_dataset = YOLODataset(val_root, get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Model setup
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 3  # Update with your actual number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Resume logic
    start_epoch = 0
    checkpoint_path = save_dir / "checkpoint_epoch_5.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    num_epochs = 10
    best_loss = float('inf')

    try:
        for epoch in range(start_epoch, num_epochs):
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

            # Save checkpoint
            current_checkpoint = save_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, train_loss, current_checkpoint)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path} with loss {best_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining paused! Current epoch will be resumed later.")
        # Save final checkpoint before exiting
        save_checkpoint(model, optimizer, epoch, train_loss, save_dir / "interrupt_checkpoint.pth")
        return

    # Save final model
    final_model_path = save_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")


if __name__ == "__main__":
    # Skip verification if resuming
    if not Path("saved_models/checkpoint_epoch_1.pth").exists():
        try:
            print("Verifying dataset...")
            ds = YOLODataset("C:/Users/LEGION/Pictures/datasets/train")
            print(f"Dataset contains {len(ds)} samples")

            img, target = ds[0]
            print("Sample 0:")
            print(f"  Image shape: {img.shape if isinstance(img, torch.Tensor) else img.size}")
            print(f"  Boxes: {len(target['boxes'])}")
            print(f"  Labels: {target['labels'].shape}")
        except Exception as e:
            print(f"\nError during dataset verification: {str(e)}")
            print("Please check your dataset paths and format before training.")
            exit()

    main()