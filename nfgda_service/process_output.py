""" Process the output of the NFGDA algorithm for a given job 
into a stack of GeoTIFFs for final display on the frontend.

Based on the projectRadarData.py script provided by Natalie. """

import numpy as np
import matplotlib.colors as mcolors
from osgeo import gdal, osr
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import os
import redis
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Radar NEXRAD Zhh colormap (cdict11 from colorlevel.py)
# ---------------------------------------------------------------------------
_cdict_nexrad_zhh = {
    'red':  ((  0.0, 150/255, 150/255),
             ( 2/19, 207/255, 207/255),
             ( 6/19,  67/255,  67/255),
             ( 7/19, 111/255, 111/255),
             ( 8/19,  53/255,  17/255),
             (11/19,   9/255,   9/255),
             (12/19,     1.0,     1.0),
             (14/19,     1.0,     1.0),
             (16/19, 113/255,     1.0),
             (17/19,     1.0,     1.0),
             (18/19, 225/255, 178/255),
             (  1.0,  99/255,  99/255)),

    'green': ((  0.0, 145/255, 145/255),
              ( 2/19, 210/255, 210/255),
              ( 6/19,  94/255,  94/255),
              ( 7/19, 214/255, 214/255),
              ( 8/19, 214/255, 214/255),
              (11/19,  94/255,  94/255),
              (12/19, 226/255, 226/255),
              (14/19, 128/255,     0.0),
              (16/19,     0.0,     1.0),
              (17/19, 146/255, 117/255),
              (18/19,     0.0,     0.0),
              (  1.0,     0.0,     0.0)),

    'blue':  ((  0.0,  83/255,  83/255),
              ( 2/19, 180/255, 180/255),
              ( 4/19, 180/255, 180/255),
              ( 6/19, 159/255, 159/255),
              ( 7/19, 232/255, 232/255),
              ( 8/19,  91/255,  24/255),
              (12/19,     0.0,     0.0),
              (16/19,     0.0,     1.0),
              (17/19,     1.0,     1.0),
              (18/19, 227/255,     1.0),
              (  1.0, 214/255, 214/255))
}

_nexrad_cmap = mcolors.LinearSegmentedColormap('radar_NEXRAD_Zhh', _cdict_nexrad_zhh)
_nexrad_boundaries = np.arange(-20, 75.1, 1)   # 96 bins, matching colorlevel.py
_nexrad_norm = mcolors.BoundaryNorm(boundaries=_nexrad_boundaries, ncolors=_nexrad_cmap.N)

# Gust-front overlay color (red, fully opaque)
_GF_RGBA = np.array([255, 0, 0, 255], dtype=np.uint8)


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

    # get radar coordinates from redis
    radar_coords = get_radar_coords(station_id, redis_client)
    if radar_coords is None:
        return f"Could not find coordinates for station {station_id} in Redis."

    # create output directory
    out_dir = f"/processed_data/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    
    radar_lon, radar_lat = radar_coords

    # process each file and collect per-frame observation timestamps
    timestamps = {}
    for i, file in enumerate(files):
        logger.info(f'processing file {i+1} of {len(files)} into GeoTIFF format')
        npz_path = os.path.join(f"/nfgda_output/{job_id}/nfgda_detection", file)
        ts = extract_timestamp(npz_path)
        if ts is not None:
            timestamps[i] = ts
        project_data(npz_path, radar_lat, radar_lon, out_dir, i)

    # Reorder frames so that indexes are in chronological order.
    timestamps = _reorder_frames_chronologically(timestamps, out_dir)

    # write manifest so the API can serve observation timestamps per frame
    manifest_path = os.path.join(out_dir, "timestamps.json")
    with open(manifest_path, "w") as f:
        json.dump(timestamps, f)
    logger.info(f"Wrote timestamp manifest with {len(timestamps)} entries to {manifest_path}")
    

def _reorder_frames_chronologically(timestamps: dict[int, str], out_dir: str) -> dict[int, str]:
    """Sort frame GeoTIFFs so that frame index 0 is the earliest """
    
    if not timestamps:
        return timestamps

    # build a list of (current_index, timestamp_str) sorted by time
    sorted_pairs = sorted(timestamps.items(), key=lambda kv: kv[1])

    # if already in order, skip the renaming process
    already_ordered = all(sorted_pairs[j][0] == j for j in range(len(sorted_pairs)))
    if already_ordered:
        logger.info("Frames are already in chronological order; skipping rename.")
        return timestamps

    logger.info("Reordering frames to chronological order: %s", [(old_idx, ts) for old_idx, ts in sorted_pairs])

    # rename every frame_<old>.tif to frame_<old>.tif.tmp
    for old_idx, _ in sorted_pairs:
        src = os.path.join(out_dir, f"frame_{old_idx}.tif")
        tmp = os.path.join(out_dir, f"frame_{old_idx}.tif.tmp")
        if os.path.exists(src):
            os.rename(src, tmp)

    # rename frame_<old>.tif.tmp to frame_<new>.tif
    new_timestamps: dict[int, str] = {}
    for new_idx, (old_idx, ts) in enumerate(sorted_pairs):
        tmp = os.path.join(out_dir, f"frame_{old_idx}.tif.tmp")
        dst = os.path.join(out_dir, f"frame_{new_idx}.tif")
        if os.path.exists(tmp):
            os.rename(tmp, dst)
        new_timestamps[new_idx] = ts

    return new_timestamps

def get_radar_coords(station_id: str, redis_client: redis.Redis) -> tuple[float, float]:
    station_json = redis_client.hget("stations", station_id)
    
    if station_json:
        station_data = json.loads(station_json)
        lon = station_data["properties"]["lon"]
        lat = station_data["properties"]["lat"]
        return (float(lon), float(lat))
    else:
        return None


def _reflectivity_to_rgba(refl: np.ndarray, nfout: np.ndarray) -> np.ndarray:
    """Convert a 2-D reflectivity array + boolean gust-front mask to RGBA uint8.

    * Valid reflectivity pixels are colored with the NEXRAD Zhh colormap.
    * NaN pixels become fully transparent (alpha = 0).
    * Gust-front detections (`nfout == True`) are drawn as dark pixels.
    """
    ny, nx = refl.shape
    rgba = np.zeros((ny, nx, 4), dtype=np.uint8)  # default: fully transparent

    valid = ~np.isnan(refl)

    # Map valid reflectivity through the colormap
    normalized = _nexrad_norm(refl[valid])                 # int bin indices
    mapped = (_nexrad_cmap(normalized) * 255).astype(np.uint8)  # (N, 4) RGBA
    mapped[:, 3] = 255  # 100% opacity for radar pixels

    rgba[valid] = mapped

    # Overlay gust-front detections (dilated slightly for visibility)
    if nfout is not None and np.any(nfout):
        gf_mask = binary_dilation(nfout, structure=disk(2))
        gf_draw = gf_mask & valid
        rgba[gf_draw] = _GF_RGBA

    return rgba


def extract_timestamp(npz_path: str) -> str | None:
    """Return the radar observation time from a detection .npz as an ISO 8601 UTC string,
    or None if the key is absent or unparseable."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "timestamp" not in data:
            return None
        ts = data["timestamp"]
        # numpy.datetime64 (0-d array) -> Python datetime -> ISO string
        # .item() is required to get the Python scalar; .astype(object) keeps it
        # as a 0-d ndarray which has no .strftime()
        ts_dt = ts.astype("datetime64[s]").item()
        return ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        logger.warning(f"Could not extract timestamp from {npz_path}: {e}")
        return None


def project_data(npz_path: str, radar_lat: float, radar_lon: float, out_dir: str, index: int) -> None:
    
    ae_tif = os.path.join(out_dir, f"radar_reflectivity_ae_{index}.tif")
    final_tif = os.path.join(out_dir, f"frame_{index}.tif")

    pixel_size_m = 500.0   # 500 m spacing
    channel_index = 1      # channel 1 = reflectivity (0-based)

    # load data
    data = np.load(npz_path)
    array = data['inputNF']
    nfout = data['nfout'] if 'nfout' in data else None

    # flip vertically
    array = np.flipud(array)
    if nfout is not None:
        nfout = np.flipud(nfout)

    refl = array[:, :, channel_index].astype(np.float64)
    ny, nx = refl.shape

    # log data range for debugging
    valid_mask = ~np.isnan(refl)
    nan_count = np.count_nonzero(~valid_mask)
    logger.info(
        f"Reflectivity stats: shape={refl.shape}, "
        f"NaN pixels={nan_count}/{refl.size}, "
        f"valid min={np.nanmin(refl):.2f}, "
        f"valid max={np.nanmax(refl):.2f}, "
        f"valid mean={np.nanmean(refl):.2f}"
    )
    if nfout is not None:
        logger.info(f"Gust-front pixels: {np.count_nonzero(nfout)}")

    # render to RGBA
    rgba = _reflectivity_to_rgba(refl, nfout)

    # spatial references
    # azimuthal equidistant centered on the radar
    ae_srs = osr.SpatialReference()
    ae_srs.SetAE(
        radar_lat,
        radar_lon,
        0.0,
        0.0
    )
    ae_srs.SetWellKnownGeogCS("WGS84")

    # geotransform (centered on radar)
    origin_x = -(nx / 2) * pixel_size_m
    origin_y =  (ny / 2) * pixel_size_m

    geotransform = (
        origin_x,
        pixel_size_m,
        0.0,
        origin_y,
        0.0,
        -pixel_size_m
    )

    # write RGBA Azimuthal Equidistant GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    ae_ds = driver.Create(
        ae_tif,
        nx,
        ny,
        4,               # R, G, B, A
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"]
    )

    ae_ds.SetGeoTransform(geotransform)
    ae_ds.SetProjection(ae_srs.ExportToWkt())

    for band_idx in range(4):
        band = ae_ds.GetRasterBand(band_idx + 1)
        band.WriteArray(rgba[:, :, band_idx])

    # set alpha band interpretation
    ae_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    ae_ds = None

    # reproject to EPSG:3857 for Leaflet
    warped_ds = gdal.Warp(
        "",
        ae_tif,
        dstSRS="EPSG:3857",
        resampleAlg=gdal.GRA_NearestNeighbour,
        format="MEM",
        dstAlpha=False,    # keep our existing alpha band
    )

    # Write final Cloud-Optimized GeoTIFF
    driver = gdal.GetDriverByName("COG")
    driver.CreateCopy(
        final_tif,
        warped_ds,
        options=["COMPRESS=DEFLATE", "OVERVIEWS=IGNORE_EXISTING"]
    )

    warped_ds = None
    logger.info(f"Wrote {final_tif}")

    # prune ae tile
    os.remove(ae_tif)

