import os
import json
import ee
import geemap
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# -------------------- EE AUTH --------------------
def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Using service account: {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Using persistent Earth Engine login")
        return "persistent"

ee.Initialize(**{
    "credentials": get_ee_credentials(),
    "project": "gee-project-368207",
})

# -------------------- CONFIG --------------------
GEE_IMAGE_ID = "users/chemuttjose/rwanda_2022_seasonA"
OUTPUT_DIR = "data/rwanda_2022_seasonA_tiles"
TILE_SIZE_PIXELS = 256  # Fixed tile size
RESOLUTION = 10  # Meters per pixel
CRS = "EPSG:4326"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- LOAD IMAGE --------------------
image = ee.Image(GEE_IMAGE_ID)
bounds = image.geometry().bounds().getInfo()["coordinates"][0]

# Get bounding box (in lon/lat)
min_lon = min([p[0] for p in bounds])
max_lon = max([p[0] for p in bounds])
min_lat = min([p[1] for p in bounds])
max_lat = max([p[1] for p in bounds])

# Convert TILE_SIZE_PIXELS to degrees (roughly for EPSG:4326)
tile_deg = (TILE_SIZE_PIXELS * RESOLUTION) / 111320

# -------------------- GENERATE GRID --------------------
print("Generating grid...")
tiles = []
y = min_lat
tile_id = 0
while y < max_lat:
    x = min_lon
    while x < max_lon:
        tile_geom = box(x, y, x + tile_deg, y + tile_deg)
        tiles.append((tile_id, tile_geom))
        tile_id += 1
        x += tile_deg
    y += tile_deg

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(tiles, columns=["tile_id", "geometry"], crs=CRS)

# -------------------- DOWNLOAD TILES --------------------
print(f"Downloading {len(gdf)} tiles...")
for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
    tile_geom = row.geometry
    coords = tile_geom.exterior.coords[:]
    ee_geom = ee.Geometry.Polygon(coords)

    clipped = image.clip(ee_geom)

    try:
        url = clipped.getDownloadURL({
            "region": ee_geom,
            "scale": RESOLUTION,
            "format": "GeoTIFF"
        })

        out_path = os.path.join(OUTPUT_DIR, f"tile_{row.tile_id:04d}.tif")
        geemap.download_file(url, out_path)
    except Exception as e:
        print(f"[{row.tile_id}] Skipped due to error: {e}")
