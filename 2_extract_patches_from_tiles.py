import timeit
start_time = timeit.default_timer()

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from tqdm import tqdm
import random
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

print('Libraries initialized successfully')

# --------------------- CONFIG --------------------- 
root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/'
TILE_ROOTS = {
    "Nyagatare_A2019": f"{root}data/Nyagatare_A2019_tiles",
    "Nyagatare_A2020": f"{root}data/Nyagatare_A2020_tiles",
    "Nyagatare_A2021": f"{root}data/Nyagatare_A2021_tiles",
    "Nyagatare_B2019": f"{root}data/Nyagatare_B2019_tiles",
    "Nyagatare_B2020": f"{root}data/Nyagatare_B2020_tiles",
    "Nyagatare_B2021": f"{root}data/Nyagatare_B2021_tiles",
}

SHAPEFILE_DIR = f"{root}data/shapefiles"
LABEL_COLUMN = "code"
VALID_LABELS = {0, 1, 2, 3}
OUTPUT_DIR = "data/patches"
PATCH_SIZE = 8
STRIDE = 4
IGNORE_VALUE = 255
SKIP_SINGLE_CLASS_PATCHES = False
NUM_WORKERS  = multiprocessing.cpu_count() - 10

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42
random.seed(SEED)

for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# --------------------- TILE PROCESSING FUNCTION ---------------------
def process_tile(tile_path, vector_path):
    result = defaultdict(list)
    local_distribution = Counter()
    multi_label_count = 0

    try:
        with rasterio.open(tile_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

            gdf = gpd.read_file(vector_path).to_crs(crs)
            gdf = gdf[gdf[LABEL_COLUMN].isin(VALID_LABELS)]
            label_shapes = ((geom, int(attr)) for geom, attr in zip(gdf.geometry, gdf[LABEL_COLUMN]))

            label_raster = rasterize(
                label_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=IGNORE_VALUE,
                dtype=np.uint8
            )

            for row in range(0, height - PATCH_SIZE + 1, STRIDE):
                for col in range(0, width - PATCH_SIZE + 1, STRIDE):
                    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                    image_patch = src.read(window=window)
                    label_patch = label_raster[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

                    if np.all(label_patch == IGNORE_VALUE):
                        continue

                    valid_pixels = label_patch[label_patch != IGNORE_VALUE]
                    if len(valid_pixels) == 0:
                        continue

                    unique_classes = set(valid_pixels)
                    if SKIP_SINGLE_CLASS_PATCHES and len(unique_classes) == 1:
                        continue

                    dominant_class = int(np.bincount(valid_pixels).argmax())
                    result[dominant_class].append((image_patch, label_patch))

                    for c in unique_classes:
                        local_distribution[c] += 1
                    if len(unique_classes) > 1:
                        multi_label_count += 1
    except Exception as e:
        print(f"  Skipped {os.path.basename(tile_path)}: {e}")

    return result, local_distribution, multi_label_count

# --------------------- MAIN LOOP ---------------------
print("Collecting patches from all seasons...")
class_distribution = Counter()
multi_label_patch_count = 0
class_to_patches = defaultdict(list)

for season_id, tile_dir in TILE_ROOTS.items():
    print(f"\nProcessing {season_id}")
    vector_path = os.path.join(SHAPEFILE_DIR, f"{season_id}.shp")
    if not os.path.exists(vector_path):
        print(f"[WARNING] Missing shapefile: {vector_path}")
        continue

    tile_files = [f for f in os.listdir(tile_dir) if f.endswith(".tif")]

    with ProcessPoolExecutor(max_workers= NUM_WORKERS) as executor:
        futures = []
        for tile_file in tile_files:
            tile_path = os.path.join(tile_dir, tile_file)
            futures.append(executor.submit(process_tile, tile_path, vector_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {season_id}"):
            tile_result, tile_distribution, tile_multi_label = future.result()
            for k, v in tile_result.items():
                class_to_patches[k].extend(v)
            class_distribution.update(tile_distribution)
            multi_label_patch_count += tile_multi_label

# --------------------- SPLIT BY CLASS ---------------------
print(f"\nTotal patches collected: {sum(len(p) for p in class_to_patches.values())}")
print(f"Multi-class patches: {multi_label_patch_count}")
print("Class pixel distribution across patches (not pixel count):")
for cls, count in sorted(class_distribution.items()):
    print(f"  Class {cls}: {count} patches contain this label")

balanced_splits = {"train": [], "val": [], "test": []}

for cls, patches in class_to_patches.items():
    train, temp = train_test_split(patches, train_size=SPLIT_RATIOS["train"], random_state=SEED)
    val, test = train_test_split(temp, test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]), random_state=SEED)

    balanced_splits["train"].extend(train)
    balanced_splits["val"].extend(val)
    balanced_splits["test"].extend(test)

# --------------------- SAVE PATCHES ---------------------
for split, items in balanced_splits.items():
    for i, (image, label) in enumerate(items):
        out_path = os.path.join(OUTPUT_DIR, split, f"patch_{split}_{i:05d}.npz")
        np.savez_compressed(out_path, image=image.astype(np.float32), mask=label.astype(np.uint8))

print("\nBalanced patches saved to:", OUTPUT_DIR)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)

## NO PARALELL PROCESSING
'''
import timeit
start_time = timeit.default_timer()
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from tqdm import tqdm
import random
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
print('Libraries initialized successfully')
root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/' #Nyagatare_B2025_tiles/
# --------------------- CONFIG --------------------- 
TILE_ROOTS = {
    "Nyagatare_A2019": f"{root}data/Nyagatare_A2019_tiles",
    "Nyagatare_A2020": f"{root}data/Nyagatare_A2020_tiles",
    "Nyagatare_A2021": f"{root}data/Nyagatare_A2021_tiles",
    "Nyagatare_B2019": f"{root}data/Nyagatare_B2019_tiles",
    "Nyagatare_B2020": f"{root}data/Nyagatare_B2020_tiles",
    "Nyagatare_B2021": f"{root}data/Nyagatare_B2021_tiles",
}
SHAPEFILE_DIR = f"{root}data/shapefiles"
LABEL_COLUMN = "code"
VALID_LABELS = {0, 1, 2, 3}
OUTPUT_DIR = "data/patches"
PATCH_SIZE = 8
STRIDE = 4
IGNORE_VALUE = 255
SKIP_SINGLE_CLASS_PATCHES = False

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42
random.seed(SEED)

# Create output folders
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# --------------------- INIT METRICS ---------------------
print("Collecting patches from all seasons...")
class_distribution = Counter()
multi_label_patch_count = 0
class_to_patches = defaultdict(list)

# --------------------- LOOP OVER SEASONS ---------------------
for season_id, tile_dir in TILE_ROOTS.items():
    print(f"\nProcessing {season_id}")
    vector_path = os.path.join(SHAPEFILE_DIR, f"{season_id}.shp")
    if not os.path.exists(vector_path):
        print(f"[WARNING] Missing shapefile: {vector_path}")
        continue

    gdf = gpd.read_file(vector_path)
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith(".tif")]

    for tile_file in tqdm(tile_files, desc=f"  Tiles in {tile_dir}"):
        tile_path = os.path.join(tile_dir, tile_file)

        try:
            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                transform = src.transform
                crs = src.crs

                gdf_ = gdf.to_crs(crs)
                gdf_ = gdf_[gdf_[LABEL_COLUMN].isin(VALID_LABELS)]
                label_shapes = ((geom, int(attr)) for geom, attr in zip(gdf_.geometry, gdf_[LABEL_COLUMN]))

                label_raster = rasterize(
                    label_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=IGNORE_VALUE,
                    dtype=np.uint8
                )

                for row in range(0, height - PATCH_SIZE + 1, STRIDE):
                    for col in range(0, width - PATCH_SIZE + 1, STRIDE):
                        window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                        image_patch = src.read(window=window)
                        label_patch = label_raster[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

                        if np.all(label_patch == IGNORE_VALUE):
                            continue

                        valid_pixels = label_patch[label_patch != IGNORE_VALUE]
                        if len(valid_pixels) == 0:
                            continue

                        unique_classes = set(valid_pixels)
                        if SKIP_SINGLE_CLASS_PATCHES and len(unique_classes) == 1:
                            continue

                        dominant_class = int(np.bincount(valid_pixels).argmax())
                        class_to_patches[dominant_class].append((image_patch, label_patch))

                        for c in unique_classes:
                            class_distribution[c] += 1
                        if len(unique_classes) > 1:
                            multi_label_patch_count += 1

        except Exception as e:
            print(f"  Skipped {tile_file}: {e}")

# --------------------- SPLIT BY CLASS ---------------------
print(f"\nTotal patches collected: {sum(len(p) for p in class_to_patches.values())}")
print(f"Multi-class patches: {multi_label_patch_count}")
print("Class pixel distribution across patches (not pixel count):")
for cls, count in sorted(class_distribution.items()):
    print(f"  Class {cls}: {count} patches contain this label")

balanced_splits = {"train": [], "val": [], "test": []}

for cls, patches in class_to_patches.items():
    train, temp = train_test_split(patches, train_size=SPLIT_RATIOS["train"], random_state=SEED)
    val, test = train_test_split(temp, test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]), random_state=SEED)

    balanced_splits["train"].extend(train)
    balanced_splits["val"].extend(val)
    balanced_splits["test"].extend(test)

# --------------------- SAVE PATCHES ---------------------
for split, items in balanced_splits.items():
    for i, (image, label) in enumerate(items):
        out_path = os.path.join(OUTPUT_DIR, split, f"patch_{split}_{i:05d}.npz")
        np.savez_compressed(out_path, image=image.astype(np.float32), mask=label.astype(np.uint8))

print("\nBalanced patches saved to:", OUTPUT_DIR)

print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
'''
