import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from collections import defaultdict
from pathlib import Path

# 1. SETUP (modify these paths)
MODEL_PATH = "saved_models/final_model.pth"
DATA_ROOT = "C:/Users/LEGION/Pictures/datasets/val"
NUM_CLASSES = 3  # Background + your classes


# 2. LOAD MODEL (with security fix)
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Security fix
    return model


# 3. ROBUST EVALUATION FUNCTION
def evaluate(model, data_loader, device):
    model.eval()
    metrics = {
        'class_1': {'tp': 0, 'fp': 0, 'fn': 0},
        'class_2': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    skipped = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Skip if no ground truth
                if len(target['boxes']) == 0:
                    skipped += 1
                    continue

                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                pred_scores = output['scores']

                # Initialize per-image counts
                img_counts = {
                    'class_1': {'tp': 0, 'fp': 0, 'fn': 0},
                    'class_2': {'tp': 0, 'fp': 0, 'fn': 0}
                }

                # Count ground truths per class
                for class_id in [1, 2]:
                    gt_count = (gt_labels == class_id).sum().item()
                    img_counts[f'class_{class_id}']['fn'] = gt_count

                # Match predictions to ground truth
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou_matrix = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()

                    for pred_idx in range(len(pred_boxes)):
                        class_id = pred_labels[pred_idx].item()
                        if class_id not in [1, 2]:
                            continue

                        # Find best matching GT
                        max_iou, gt_idx = iou_matrix[pred_idx].max(dim=0)
                        gt_idx = gt_idx.item()

                        if max_iou >= 0.5 and gt_labels[gt_idx] == class_id and gt_idx not in matched_gt:
                            img_counts[f'class_{class_id}']['tp'] += 1
                            img_counts[f'class_{class_id}']['fn'] -= 1
                            matched_gt.add(gt_idx)
                        else:
                            img_counts[f'class_{class_id}']['fp'] += 1

                # Accumulate counts
                for class_id in [1, 2]:
                    for metric in ['tp', 'fp', 'fn']:
                        metrics[f'class_{class_id}'][metric] += img_counts[f'class_{class_id}'][metric]

    print(f"\nSkipped {skipped} invalid samples")

    # Calculate final metrics
    results = {}
    for class_id in [1, 2]:
        key = f'class_{class_id}'
        tp = metrics[key]['tp']
        fp = metrics[key]['fp']
        fn = metrics[key]['fn']

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        results[f'{key}_precision'] = precision
        results[f'{key}_recall'] = recall
        results[f'{key}_f1'] = 2 * (precision * recall) / (precision + recall + 1e-6)

    return results


# 4. MAIN EXECUTION
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model().to(device)

    # Prepare data
    from dataset import YOLODataset

    val_dataset = YOLODataset(DATA_ROOT, transforms=T.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=lambda x: tuple(zip(*x)))

    # Run evaluation
    metrics = evaluate(model, val_loader, device)

    # Print results
    print("\n=== MODEL EVALUATION METRICS ===")
    print(f"{'Metric':<25} | {'Value':>10}")
    print("-" * 38)
    for metric, value in metrics.items():
        print(f"{metric:<25} | {value:>10.4f}")