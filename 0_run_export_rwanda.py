# run_export_rwanda.py
import timeit
start_time = timeit.default_timer()
from datetime import date
import geopandas as gpd
from src.data.earthengine.eo import EarthEngineExporter

print('Necessary libraries imported')

# Parameters
root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/'
eyear = 2019
season = 'B'
start = date(eyear, 2, 1)
end = date(eyear, 6, 30)

# Load the shapefile
gdf = gpd.read_file(f"{root}data/rwa_adm2_selected_districts.shp")

# Filter for 'Nyagatare' polygon
nyagatare_gdf = gdf[gdf['ADM2_EN'] == 'Nyagatare']

if nyagatare_gdf.empty:
    raise ValueError("Nyagatare not found in ADM2_EN column.")

# Extract geometry in GeoJSON format
aoi_geojson_geometry = nyagatare_gdf.iloc[0].geometry.__geo_interface__

# Initialize Earth Engine exporter
exporter = EarthEngineExporter(mode="batch")

# Export data for Nyagatare
exporter.export_for_geo_json(
    geo_json=aoi_geojson_geometry,
    start_date=start,
    end_date=end,
    identifier=f"Rwanda_{season}{eyear}"
)

print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)


'''
from datetime import date
import json
from src.data.earthengine.eo import EarthEngineExporter


with open("data/nyangatare.geojson") as f:
    aoi_geojson = json.load(f)

exporter = EarthEngineExporter(
    mode="batch", 
)

exporter.export_for_geo_json(
    geo_json=aoi_geojson["features"][0]["geometry"],
    start_date=date(2021, 9, 1),
    end_date=date(2022, 2, 28),
    identifier="rwanda_2022_seasonA"
)
'''