""" Process the output of the NFGDA algorithm for a given job 
into a stack of GeoTIFFs for final display on the frontend.

Based on the projectRadarData.py script provided by Natalie. """

import numpy as np
from osgeo import gdal, osr
import os
import redis
import json
import logging

logger = logging.getLogger(__name__)

def generate_geotiff_output(job_id: str, redis_client: redis.Redis):

    # get station id from redis
    station_id = redis_client.hget(f"job:{job_id}", "stationId")
    if station_id is None:
        return f"Could not find station ID for job {job_id} in Redis."
    
    # check output presence and return list of files
    if not os.path.exists(f"/nfgda_output/{job_id}/nfgda_detection"):
        return "Could not find output directory"
    
    # get list of files
    files = os.listdir(f"/nfgda_output/{job_id}/nfgda_detection")
    files = [f for f in files if f.endswith(".npz")]
    if len(files) == 0:
        return "No files found in output directory"

    # create output directory
    out_dir = f"/processed_data/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)

    # get radar coordinates from redis
    radar_coords = get_radar_coords(station_id, redis_client)
    if radar_coords is None:
        return f"Could not find coordinates for station {station_id} in Redis."
    
    radar_lon, radar_lat = radar_coords

    # process each file
    for i, file in enumerate(files):
        logger.info(f'processing file {i+1} of {len(files)} into GeoTIFF format')
        project_data(os.path.join(f"/nfgda_output/{job_id}/nfgda_detection", file), radar_lat, radar_lon, out_dir, i)
    

def get_radar_coords(station_id: str, redis_client: redis.Redis) -> tuple[float, float]:
    station_json = redis_client.hget("stations", station_id)
    
    if station_json:
        station_data = json.loads(station_json)
        lon = station_data["properties"]["lon"]
        lat = station_data["properties"]["lat"]
        return (float(lon), float(lat))
    else:
        return None

def project_data(npz_path: str, radar_lat: float, radar_lon: float, out_dir: str, index:int) -> None:

    # ---------------------------
    # Other parameters
    # ---------------------------
    ae_tif = os.path.join(out_dir, f"radar_reflectivity_ae_{index}.tif")
    final_tif = os.path.join(out_dir, f"radar_reflectivity_3857_{index}.tif")

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
    logger.info(ae_srs.ExportToPrettyWkt())

    # ---------------------------
    # GeoTransform (centered on radar)
    # ---------------------------
    origin_x = - (nx / 2) * pixel_size_m
    origin_y = (ny / 2) * pixel_size_m
    logger.info(f"origin_x, origin_y = {origin_x, origin_y}")

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
        logger.warning("No nodata value defined on band.")
    else:
        remaining = np.count_nonzero(arr == nodata)

    if remaining == 0:
        logger.info("Success: No nodata pixels remain in band.")
    else:
        logger.warning(f"Warning: {remaining} nodata pixels still present. Try increasing maxSearchDist.")

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

  

    

