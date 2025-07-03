# run_export_rwanda.py

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
