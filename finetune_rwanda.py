import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PixelwisePatchDataset
from src.galileo import Encoder
from src.data.utils import construct_galileo_input


class PixelwisePatchClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.classifier = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=num_classes,
            kernel_size=1
        )

    def encode_features(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 6, 12, H, W).permute(0, 1, 3, 4, 2).contiguous()
        inputs = []

        for b in range(B):
            s1 = x[b, ..., :2].permute(1, 2, 0, 3).float()
            s2 = x[b, ..., 2:].permute(1, 2, 0, 3).float()
            masked = construct_galileo_input(s1=s1, s2=s2, normalize=True)
            inputs.append(masked)

        batched_input = {
            k: torch.stack([getattr(i, k).float() if k != "months" else getattr(i, k).long() for i in inputs])
            for k in inputs[0]._fields
        }

        feats, *_ = self.encoder(
            batched_input["space_time_x"],
            batched_input["space_x"],
            batched_input["time_x"],
            batched_input["static_x"],
            batched_input["space_time_mask"],
            batched_input["space_mask"],
            batched_input["time_mask"],
            batched_input["static_mask"],
            batched_input["months"],
            patch_size=H,
        )
        return feats

    def forward(self, x):
        feats = self.encode_features(x)
        while feats.dim() > 5:
            feats = feats.squeeze(1)
        feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return self.classifier(feats)  # [B, num_classes, H, W]


def compute_class_weights(dataset, num_classes):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_counts = torch.zeros(num_classes)

    for _, mask in loader:
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()

    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    return weights


def compute_mIoU(preds, targets, num_classes, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        iou = intersection / union
        ious.append(iou)
    return sum(ious) / len(ious) if ious else 0.0


def train(args):
    print(f"[INFO] Loading datasets from: {args.data_dir}")
    train_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="train")
    val_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="val")

    num_classes = train_dataset.num_classes
    print(f"[INFO] Number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder.load_from_folder(Path(args.encoder_ckpt))
    model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=args.freeze_encoder).to(args.device)

    weights = compute_class_weights(train_dataset, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, mask in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            x, mask = x.to(args.device), mask.to(args.device)

            optimizer.zero_grad()
            logits = model(x)
            logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)

            loss = criterion(logits, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[WARN] Skipping batch due to invalid loss.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += ((preds == mask) & (mask != 255)).sum().item()
            total += (mask != 255).sum().item()

        train_acc = correct / total
        print(f"[Epoch {epoch}] Train Loss = {total_loss:.4f}, Accuracy = {train_acc:.4f}")

        # Validation with mIoU
        model.eval()
        total_miou = 0.0
        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(args.device), mask.to(args.device)
                logits = model(x)
                logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)
                total_miou += compute_mIoU(preds, mask, num_classes)

        mean_miou = total_miou / len(val_loader)
        print(f"[Epoch {epoch}] Train Loss = {total_loss:.4f}, Train Acc = {train_acc:.4f}, Val mIoU = {mean_miou:.4f}")

        if mean_miou > best_val_miou:
            best_val_miou = mean_miou
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            print(f"[INFO] Best model saved based on mIoU = {mean_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="pixelwise_checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    train(args)
