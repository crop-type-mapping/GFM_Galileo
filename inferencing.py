import os
import torch
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from rasterio.merge import merge

from src.galileo import Encoder
from src.data.utils import construct_galileo_input
from pixel_wise_train_classifier import PixelwisePatchClassifier

# --- SETTINGS ---
tile_folder = Path("data/input_tiles/")
output_folder = Path("output/output_tiles/")
output_folder.mkdir(parents=True, exist_ok=True)
final_output_path = "output/merged_prediction.tif"

encoder_ckpt = "data/models/nano/"
model_ckpt = "checkpoints/best_model.pt"

patch_size = 8
stride = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 4
ignore_value = 255
confidence_threshold = 0.5

# --- Load model ---
print("[INFO] Loading encoder and model")
encoder = Encoder.load_from_folder(Path(encoder_ckpt))
model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=True)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model = model.to(device).eval()

# --- Process each tile ---
tile_paths = sorted(tile_folder.glob("*.tif"))
pred_tiles = []

print(f"[INFO] Found {len(tile_paths)} tiles.")

for tile_path in tqdm(tile_paths, desc="Processing tiles"):
    print(f"[DEBUG] Processing tile: {tile_path.name}")
    with rasterio.open(tile_path) as src:
        image = src.read()  # [C=60, H, W]
        profile = src.profile
        H, W = image.shape[1:]

        print(f"[DEBUG] Tile shape: {image.shape}")
        if image.shape[0] != 60:
            raise ValueError(f"[ERROR] Tile must have 60 channels (5 timesteps x 12 bands). Got {image.shape[0]}")

        # Pad tile if needed
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        H_pad, W_pad = image_padded.shape[1:]
        print(f"[DEBUG] Padded tile shape: {image_padded.shape}")

        pred_map = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
        count_map = np.zeros((H_pad, W_pad), dtype=np.uint8)

        # Slide window and infer
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image_padded[:, y:y+patch_size, x:x+patch_size]  # [60, 8, 8]

                patch = patch.reshape(6, 12, patch_size, patch_size).transpose(0, 2, 3, 1)
                s1 = patch[..., :2].transpose(1, 2, 0, 3)  # [H, W, T, 2]
                s2 = patch[..., 2:].transpose(1, 2, 0, 3)  # [H, W, T, 10]

                s1 = torch.from_numpy(s1).float()
                s2 = torch.from_numpy(s2).float()

                masked = construct_galileo_input(s1=s1, s2=s2, normalize=True)
                batched_input = {
                    k: torch.stack([getattr(masked, k).float() if k != "months" else getattr(masked, k).long()])
                    for k in masked._fields
                }
                batched_input = {k: v.to(device) for k, v in batched_input.items()}

                with torch.no_grad():
                    feats, *_ = encoder(
                        batched_input["space_time_x"],
                        batched_input["space_x"],
                        batched_input["time_x"],
                        batched_input["static_x"],
                        batched_input["space_time_mask"],
                        batched_input["space_mask"],
                        batched_input["time_mask"],
                        batched_input["static_mask"],
                        batched_input["months"],
                        patch_size=patch_size,
                    )
                    feats = feats.squeeze(1)[:, -1, :, :, :]  # [1, C, H, W]
                    feats = feats.permute(0, 3, 1, 2).contiguous()
                    logits = model.classifier(feats)

                    probs = torch.softmax(logits, dim=1)
                    probs = F.interpolate(probs, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    probs = probs.squeeze(0).cpu().numpy()  # [num_classes, 8, 8]

                pred_map[:, y:y+patch_size, x:x+patch_size] += probs
                count_map[y:y+patch_size, x:x+patch_size] += 1

        # --- Generate final prediction mask ---
        avg_probs = pred_map / np.clip(count_map, a_min=1, a_max=None)
        confidence = np.max(avg_probs, axis=0)
        final_mask = np.argmax(avg_probs, axis=0).astype(np.uint8)

        # Create valid data mask from original (unpadded) image
        valid_data_mask = np.isfinite(image).all(axis=0) & (np.abs(image).sum(axis=0) >= 1e-6)
        valid_data_mask_padded = np.pad(valid_data_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)

        # Combine conditions: low confidence, no overlapping patches, or invalid input
        mask_low_conf = (confidence < confidence_threshold) | (count_map == 0) | (~valid_data_mask_padded)
        final_mask[mask_low_conf] = ignore_value

        # Remove padding
        final_mask = final_mask[:H, :W]

        ignore_ratio = (final_mask == ignore_value).sum() / final_mask.size
        
        # --- Save prediction mask ---
        out_tile_path = output_folder / tile_path.name.replace(".tif", "_pred.tif")
        profile.update(count=1, dtype='uint8', height=H, width=W)
        with rasterio.open(out_tile_path, "w", **profile) as dst:
            dst.write(final_mask, 1)

        pred_tiles.append(out_tile_path)

# --- Merge predicted tiles ---
srcs = [rasterio.open(p) for p in pred_tiles]
mosaic, out_transform = merge(srcs)

profile = srcs[0].profile
profile.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
    "count": 1,
    "dtype": "uint8"
})

with rasterio.open(final_output_path, "w", **profile) as dst:
    dst.write(mosaic.astype(np.uint8))

print(f"[INFO] Merged prediction saved to: {final_output_path}")
