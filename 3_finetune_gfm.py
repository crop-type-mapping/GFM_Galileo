import timeit
start_time = timeit.default_timer()
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

from src.data.dataset import PixelwisePatchDataset
from src.galileo import Encoder
from src.data.utils import construct_galileo_input

nsteps = 5 # number of months or steps
nbands = 12 # Number of bands
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
        x = x.view(B, nsteps, nbands, H, W).permute(0, 1, 3, 4, 2).contiguous()
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
        while feats.dim() > nsteps:
            feats = feats.squeeze(1)
        feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return self.classifier(feats)


def compute_class_weights(dataset, num_classes):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_counts = torch.zeros(num_classes)

    for _, mask in loader:
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()

    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    return weights


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_miou = 0.0

    val_miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(args.device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += ((preds == mask) & (mask != 255)).sum().item()
            total += (mask != 255).sum().item()

        train_acc = correct / total
        model.eval()
        val_correct, val_total = 0, 0
        val_miou_metric.reset()

        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(args.device), mask.to(args.device)
                logits = model(x)
                logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)
                val_correct += ((preds == mask) & (mask != 255)).sum().item()
                val_total += (mask != 255).sum().item()
                val_miou_metric.update(preds, mask)

        val_acc = val_correct / val_total
        mean_miou = val_miou_metric.compute().item()
        scheduler.step(mean_miou)

        print(f"[Epoch {epoch}] LR: {optimizer.param_groups[0]['lr']:.6f}, Train Loss = {total_loss:.4f}, "
              f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val mIoU = {mean_miou:.4f}")

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
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
