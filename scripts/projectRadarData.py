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
ae_tif = "radar_reflectivity_ae.tif"
final_tif = "radar_reflectivity_3857.tif"

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
# Spatial references
# use Azimuthal Equidistant for radar data
# no EPSG code, parameterize instead
# ---------------------------
ae_srs = osr.SpatialReference()
ae_srs.SetAE(
    radar_lat,   # latitude of projection center
    radar_lon,   # longitude of projection center
    0.0,         # false easting
    0.0          # false northing
)
ae_srs.SetWellKnownGeogCS("WGS84")
print(ae_srs.ExportToPrettyWkt())

# ---------------------------
# GeoTransform (centered on radar)
# ---------------------------
origin_x = - (nx / 2) * pixel_size_m
origin_y = (ny / 2) * pixel_size_m
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
# Write Azmithual Equidistant GeoTIFF
# ---------------------------
driver = gdal.GetDriverByName("GTiff")
ae_ds = driver.Create(
    ae_tif,
    nx,
    ny,
    1,
    gdal.GDT_Float64,
    options=["COMPRESS=LZW", "TILED=YES"]
)

ae_ds.SetGeoTransform(geotransform)
ae_ds.SetProjection(ae_srs.ExportToWkt())

band = ae_ds.GetRasterBand(1)
band.WriteArray(refl)
band.SetNoDataValue(-9999)

ae_ds = None

# ---------------------------
# Reproject to EPSG:3857 for leaflet
# ---------------------------
warped_ds = gdal.Warp(
    "",
    ae_tif,
    dstSRS="EPSG:3857",
    resampleAlg=gdal.GRA_NearestNeighbour,
    format="MEM",
    dstNodata=-9999
)

# ---------------------------
# Fill nodata collar in EPSG:3857 output
# ---------------------------
band = warped_ds.GetRasterBand(1)

nodata = band.GetNoDataValue()
if nodata is None:
    nodata = -9999
    band.SetNoDataValue(nodata)

# Expand nearest valid pixels into nodata areas
gdal.FillNodata(
    targetBand=band,
    maskBand=None,
    maxSearchDist=200,        # pixels to search outward (adjust if needed)
    smoothingIterations=0     # 0 keeps nearest-neighbor style behavior
)

band.FlushCache()
warped_ds.FlushCache()

# ---------------------------
# Verify no nodata remains
# ---------------------------
arr = band.ReadAsArray()
nodata = band.GetNoDataValue()

if nodata is None:
    print("Warning: No nodata value defined on band.")
else:
    remaining = np.count_nonzero(arr == nodata)

if remaining == 0:
    print("Success: No nodata pixels remain in band.")
else:
    print(f"Warning: {remaining} nodata pixels still present. Try increasing maxSearchDist.")

# ---------------------------
# Write final GeoTIFF
# ---------------------------
driver = gdal.GetDriverByName("GTiff")
driver.CreateCopy(
    final_tif,
    warped_ds,
    options=["COMPRESS=LZW", "TILED=YES"]
)

warped_ds = None

print("Done.")
print(f"Intermediate Azimuthal Equidistant GeoTIFF: {ae_tif} (EPSG:54032)")
print(f"Final lat/lon GeoTIFF: {final_tif} (EPSG:3857)")
