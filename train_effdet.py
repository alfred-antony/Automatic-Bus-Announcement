import torch
import os
from effdet import create_model, get_efficientdet_config
from effdet.config import get_efficientdet_config
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from PIL import Image
import torchvision.transforms as T
import time
import numpy as np


class CocoDataset(Dataset):
    def __init__(self, root, ann_file, img_size=512, is_training=True):
        self.root = root
        self.img_size = img_size
        self.is_training = is_training

        with open(ann_file) as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(sorted(self.images.keys()))
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img = self.transform(img)

        # Get annotations
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return img, target


class EfficientDetTrainer:
    def __init__(self, model, train_loader, val_loader, device, num_epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs)
        self.best_map = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        last_print = time.time()

        for i, (images, targets) in enumerate(self.train_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()

            running_loss += losses.item()

            if time.time() - last_print > 30 or i == len(self.train_loader) - 1:
                print(f'Epoch: {epoch} | Batch: {i + 1}/{len(self.train_loader)} | Loss: {losses.item():.4f}')
                last_print = time.time()

        self.scheduler.step()
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        # Implement your validation metrics here
        # For simplicity, we'll just return 0
        return {'map': 0.0, 'map50': 0.0, 'map75': 0.0}

    def fit(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val mAP: {val_metrics["map"]:.4f}')
            print(f'Val mAP50: {val_metrics["map50"]:.4f}')
            print(f'Val mAP75: {val_metrics["map75"]:.4f}\n')


def collate_fn(batch):
    return tuple(zip(*batch))


def train_efficientdet():
    # Configuration
    model_name = 'tf_efficientdet_d1'
    num_classes = 2  # Update this with your number of classes
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.01
    image_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths - UPDATE THESE
    data_dir = "C:/Users/LEGION/Pictures/datasets_coco/"
    train_images_dir = os.path.join(data_dir, "train")
    val_images_dir = os.path.join(data_dir, "val")
    train_json_path = os.path.join(data_dir, "annotations/train_coco.json")
    val_json_path = os.path.join(data_dir, "annotations/val_coco.json")

    # Create model
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    model = create_model(
        model_name,
        bench_task='train',
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    # Create datasets
    train_dataset = CocoDataset(
        root=train_images_dir,
        ann_file=train_json_path,
        img_size=image_size,
        is_training=True
    )

    val_dataset = CocoDataset(
        root=val_images_dir,
        ann_file=val_json_path,
        img_size=image_size,
        is_training=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Train
    trainer = EfficientDetTrainer(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs,
        learning_rate
    )
    trainer.fit()

    # Save model
    model_save_path = os.path.join(data_dir, 'efficientdet_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    train_efficientdet()