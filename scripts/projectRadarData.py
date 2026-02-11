#!/usr/bin/env python3

import argparse
import numpy as np
from osgeo import gdal, osr
import math

# ---------------------------
# Command line inputs
# ---------------------------
parser = argparse.ArgumentParser(
    description="Convert radar NPZ data to GeoTIFF using radar lat/lon."
)

parser.add_argument(
    "npz_path",
    type=str,
    help="Path to input NPZ file"
)

parser.add_argument(
    "radar_lat",
    type=float,
    help="Radar latitude in degrees"
)

parser.add_argument(
    "radar_lon",
    type=float,
    help="Radar longitude in degrees"
)

args = parser.parse_args()

npz_path = args.npz_path
radar_lat = args.radar_lat
radar_lon = args.radar_lon

#------------example input data for testing:
#npz_path = "nf_predKABX20200707_012805_V06.npz"
#KABX radar location
#radar_lat = 35.149722    # degrees
#radar_lon = -106.823889  # degrees

# ---------------------------
# Other parameters
# ---------------------------
utm_tif = "radar_reflectivity_utm.tif"
final_tif = "radar_reflectivity_latlon.tif"

pixel_size_m = 500.0   # 500 m spacing
channel_index = 1      # channel 1 (0-based)

# ---------------------------
# Load data
# ---------------------------
data = np.load(npz_path)
array = data['inputNF'] 

# Flip vertically (upside down)
array = np.flipud(array)

refl = array[:, :, channel_index].astype(np.float64)
ny, nx = refl.shape
print(f"ny, nx = {ny, nx}")

# ---------------------------
# Determine best-fit UTM zone
# ---------------------------
utm_zone = int((radar_lon + 180) / 6) + 1
print(f"utm_zone = {utm_zone}")
is_northern = radar_lat >= 0
print(f"is_northern = {is_northern}")
epsg_utm = 32600 + utm_zone if is_northern else 32700 + utm_zone
print(f"epsg_utm = {epsg_utm}")

# ---------------------------
# Spatial references
# ---------------------------
utm_srs = osr.SpatialReference()
utm_srs.ImportFromEPSG(epsg_utm)
print(f"utm_srs = {utm_srs}")

wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)

# ---------------------------
# Transform radar center to UTM
# ---------------------------
to_utm = osr.CoordinateTransformation(wgs84, utm_srs)
center_x, center_y, _ = to_utm.TransformPoint(radar_lat, radar_lon)
print(f"center_x, center_y = {center_x, center_y}")

# ---------------------------
# GeoTransform (centered grid)
# ---------------------------
origin_x = center_x - (nx / 2) * pixel_size_m
origin_y = center_y + (ny / 2) * pixel_size_m
print(f"origin_x, origin_y = {origin_x, origin_y}")

geotransform = (
    origin_x,
    pixel_size_m,
    0.0,
    origin_y,
    0.0,
    -pixel_size_m
)

# ---------------------------
# Write UTM GeoTIFF
# ---------------------------
driver = gdal.GetDriverByName("GTiff")
utm_ds = driver.Create(
    utm_tif,
    nx,
    ny,
    1,
    gdal.GDT_Float64,
    options=["COMPRESS=LZW", "TILED=YES"]
)

utm_ds.SetGeoTransform(geotransform)
utm_ds.SetProjection(utm_srs.ExportToWkt())

band = utm_ds.GetRasterBand(1)
band.WriteArray(refl)
band.SetNoDataValue(-9999)

band.FlushCache()
utm_ds.FlushCache()
utm_ds = None

# ---------------------------
# Reproject to EPSG:4326
# ---------------------------
gdal.Warp(
    final_tif,
    utm_tif,
    dstSRS="EPSG:4326",
    resampleAlg=gdal.GRA_Bilinear,
    format="GTiff",
    creationOptions=["COMPRESS=LZW", "TILED=YES"]
)

print("Done.")
print(f"Intermediate UTM GeoTIFF: {utm_tif} (EPSG:{epsg_utm})")
print(f"Final lat/lon GeoTIFF:   {final_tif} (EPSG:4326)")
