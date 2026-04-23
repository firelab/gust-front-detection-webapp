""" Process the output of the NFGDA algorithm for a given job 
into a stack of GeoTIFFs for final display on the frontend.

Based on the projectRadarData.py script provided by Natalie. """

import numpy as np
import matplotlib.colors as mcolors
from osgeo import gdal, osr
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize, disk

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

# Gust-front overlay colors
_GF_RGBA_DETECTED       = np.array([255,   0,   0, 255], dtype=np.uint8)  # red             – observed detection
_GF_RGBA_FC_30          = np.array([255, 140,   0, 255], dtype=np.uint8)  # orange          – forecast ≥ 30%
_GF_RGBA_FC_50          = np.array([220,  10, 120, 255], dtype=np.uint8)  # reddish-fuchsia – forecast ≥ 50% (slightly dimmed)
_GF_RGBA_FC_75          = np.array([120,   0, 255, 255], dtype=np.uint8)  # vibrant violet  – forecast ≥ 75%
_GF_RGBA_FORECAST       = _GF_RGBA_FC_75   # alias used in detection rendering (is_forecast=True path)


def generate_geotiff_output(job_id: str, redis_client: redis.Redis):

    # get station id from redis
    station_id = redis_client.hget(f"job:{job_id}", "stationId")
    if station_id is None:
        return f"Could not find station ID for job {job_id} in Redis."
    
    # check output presence and return list of files
    if not os.path.exists(f"/nfgda_output/{job_id}/nfgda_detection"):
        return "Could not find output directory"
    
    # get list of detection files
    det_files = os.listdir(f"/nfgda_output/{job_id}/nfgda_detection")
    det_files = [f for f in det_files if f.endswith(".npz")]
    if len(det_files) == 0:
        return "No files found in output directory"

    # get radar coordinates from redis
    radar_coords = get_radar_coords(station_id, redis_client)
    if radar_coords is None:
        return f"Could not find coordinates for station {station_id} in Redis."

    # create output directory
    out_dir = f"/processed_data/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    
    radar_lon, radar_lat = radar_coords

    # ---------------------------------------------------------------
    # Pass 1: process observed detection frames
    # ---------------------------------------------------------------
    # manifest entries: { frame_index: { "timestamp": "...", "is_forecast": bool } }
    manifest: dict[int, dict] = {}

    # also track each detection file's timestamp so we can find the latest one
    det_ts_to_npz: dict[str, str] = {}
    for i, file in enumerate(det_files):
        logger.info(f'processing detection file {i+1} of {len(det_files)} into GeoTIFF format')
        npz_path = os.path.join(f"/nfgda_output/{job_id}/nfgda_detection", file)
        ts = extract_timestamp(npz_path)
        if ts is not None:
            manifest[i] = {"timestamp": ts, "is_forecast": False}
            det_ts_to_npz[ts] = npz_path
        project_data(npz_path, radar_lat, radar_lon, out_dir, i, is_forecast=False)

    # reorder detection frames so indexes are in chronological order
    manifest = _reorder_frames_chronologically(manifest, out_dir)
    num_detection_frames = len(manifest)

    # find the npz path for the latest detection frame (used as frozen background in forecasts)
    last_det_npz: str | None = None
    if manifest:
        last_det_ts_str = manifest[max(manifest.keys())]["timestamp"]
        last_det_npz = det_ts_to_npz.get(last_det_ts_str)
        if last_det_npz:
            logger.info(f"Last detection frame npz: {os.path.basename(last_det_npz)} ({last_det_ts_str})")

    # ---------------------------------------------------------------
    # Pass 2: process forecast frames from forecast-summary/
    # Only include frames whose valid time falls within [last_detection_ts,
    # last_detection_ts + 1 hour).  Frames with earlier timestamps are
    # redundant (covered by actual detections), and frames beyond one hour
    # are too speculative to display.
    # ---------------------------------------------------------------
    forecast_dir = f"/nfgda_output/{job_id}/forecast-summary"
    forecast_files = []
    if os.path.exists(forecast_dir):
        # derive the last detection timestamp from the reordered manifest
        last_det_ts: np.datetime64 | None = None
        if manifest:
            last_det_entry = manifest[max(manifest.keys())]
            try:
                last_det_ts = np.datetime64(last_det_entry["timestamp"].rstrip("Z"), "s")
            except Exception:
                pass

        if last_det_ts is not None:
            window_end = last_det_ts + np.timedelta64(3600, "s")  # +1 hour
            candidate_files = sorted([
                f for f in os.listdir(forecast_dir) if f.endswith(".npz")
            ])
            for fname in candidate_files:
                npz_path = os.path.join(forecast_dir, fname)
                ts_str = extract_timestamp(npz_path)
                if ts_str is None:
                    continue
                try:
                    fc_ts = np.datetime64(ts_str.rstrip("Z"), "s")
                except Exception:
                    continue
                if last_det_ts <= fc_ts <= window_end:
                    forecast_files.append(fname)
            logger.info(
                f"Forecast window: [{last_det_ts}Z, {window_end}Z] — "
                f"{len(forecast_files)} of {len(candidate_files)} file(s) selected"
            )
        else:
            logger.warning("Could not determine last detection timestamp; skipping forecast pass")

    if forecast_files:
        logger.info(f"Processing {len(forecast_files)} forecast frame(s)")
        for j, file in enumerate(forecast_files):
            frame_idx = num_detection_frames + j
            logger.info(f'processing forecast file {j+1} of {len(forecast_files)} into GeoTIFF format')
            npz_path = os.path.join(forecast_dir, file)
            ts = extract_timestamp(npz_path)
            if ts is not None:
                manifest[frame_idx] = {"timestamp": ts, "is_forecast": True}
            project_forecast(npz_path, radar_lat, radar_lon, out_dir, frame_idx,
                             background_det_npz=last_det_npz)
    else:
        logger.info("No forecast-summary frames found in window; skipping forecast pass")

    # write manifest so the API can serve per-frame metadata
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    logger.info(f"Wrote manifest with {len(manifest)} entries to {manifest_path}")


def _reorder_frames_chronologically(manifest: dict[int, dict], out_dir: str) -> dict[int, dict]:
    """Sort frame GeoTIFFs so that frame index 0 is the earliest observation.

    manifest maps current frame index → { timestamp, is_forecast }.
    Renames frame_<i>.tif files so that indexes are in chronological order,
    and returns a new manifest keyed by the corrected indexes.
    """
    if not manifest:
        return manifest

    # build a list of (current_index, entry) sorted by timestamp
    sorted_pairs = sorted(manifest.items(), key=lambda kv: kv[1]["timestamp"])

    # if already in order, skip the renaming process
    already_ordered = all(sorted_pairs[j][0] == j for j in range(len(sorted_pairs)))
    if already_ordered:
        logger.info("Frames are already in chronological order; skipping rename.")
        return manifest

    logger.info("Reordering frames to chronological order: %s",
                [(old_idx, entry["timestamp"]) for old_idx, entry in sorted_pairs])

    # rename every frame_<old>.tif to frame_<old>.tif.tmp first to avoid collisions
    for old_idx, _ in sorted_pairs:
        src = os.path.join(out_dir, f"frame_{old_idx}.tif")
        tmp = os.path.join(out_dir, f"frame_{old_idx}.tif.tmp")
        if os.path.exists(src):
            os.rename(src, tmp)

    # rename frame_<old>.tif.tmp to frame_<new>.tif
    new_manifest: dict[int, dict] = {}
    for new_idx, (old_idx, entry) in enumerate(sorted_pairs):
        tmp = os.path.join(out_dir, f"frame_{old_idx}.tif.tmp")
        dst = os.path.join(out_dir, f"frame_{new_idx}.tif")
        if os.path.exists(tmp):
            os.rename(tmp, dst)
        new_manifest[new_idx] = entry

    return new_manifest


def get_radar_coords(station_id: str, redis_client: redis.Redis) -> tuple[float, float]:
    station_json = redis_client.hget("stations", station_id)
    
    if station_json:
        station_data = json.loads(station_json)
        lon = station_data["properties"]["lon"]
        lat = station_data["properties"]["lat"]
        return (float(lon), float(lat))
    else:
        return None


def _reflectivity_to_rgba(refl: np.ndarray, nfout: np.ndarray, is_forecast: bool = False) -> np.ndarray:
    """Convert a 2-D reflectivity array + boolean gust-front mask to RGBA uint8.

    * Valid reflectivity pixels are colored with the NEXRAD Zhh colormap.
    * NaN pixels become fully transparent (alpha = 0).
    * Detected gust-fronts (is_forecast=False) are drawn red.
    * Forecast gust-fronts (is_forecast=True) are drawn bright purple.
    """
    ny, nx = refl.shape
    rgba = np.zeros((ny, nx, 4), dtype=np.uint8)  # default: fully transparent

    valid = ~np.isnan(refl)

    # Map valid reflectivity through the colormap
    normalized = _nexrad_norm(refl[valid])                 # int bin indices
    mapped = (_nexrad_cmap(normalized) * 255).astype(np.uint8)  # (N, 4) RGBA
    mapped[:, 3] = 255  # 100% opacity for radar pixels

    rgba[valid] = mapped

    # Overlay gust-front skeleton pixels (disk(1) dilation for visibility)
    if nfout is not None and np.any(nfout):
        gf_color = _GF_RGBA_FORECAST if is_forecast else _GF_RGBA_DETECTED
        gf_dilated = binary_dilation(nfout.astype(bool), structure=disk(1))
        gf_draw = gf_dilated & valid
        rgba[gf_draw] = gf_color

    return rgba


def extract_timestamp(npz_path: str) -> str | None:
    """Return the observation/valid time from a .npz as an ISO 8601 UTC string,
    or None if the key is absent or unparseable.

    Works for both detection files and forecast-summary files — both store
    timestamps as numpy.datetime64 scalars.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "timestamp" not in data:
            return None
        ts = data["timestamp"]
        ts_dt = ts.astype("datetime64[s]").item()
        return ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        logger.warning(f"Could not extract timestamp from {npz_path}: {e}")
        return None


def _write_geotiff(rgba: np.ndarray, radar_lat: float, radar_lon: float,
                   out_dir: str, index: int) -> None:
    """Shared GeoTIFF writing logic: AE projection → reproject to EPSG:3857 → COG."""
    ae_tif   = os.path.join(out_dir, f"radar_reflectivity_ae_{index}.tif")
    final_tif = os.path.join(out_dir, f"frame_{index}.tif")

    pixel_size_m = 500.0
    ny, nx = rgba.shape[:2]

    # spatial references – azimuthal equidistant centered on the radar
    ae_srs = osr.SpatialReference()
    ae_srs.SetAE(radar_lat, radar_lon, 0.0, 0.0)
    ae_srs.SetWellKnownGeogCS("WGS84")

    # geotransform (centered on radar)
    origin_x = -(nx / 2) * pixel_size_m
    origin_y =  (ny / 2) * pixel_size_m
    geotransform = (origin_x, pixel_size_m, 0.0, origin_y, 0.0, -pixel_size_m)

    # write RGBA Azimuthal Equidistant GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    ae_ds = driver.Create(ae_tif, nx, ny, 4, gdal.GDT_Byte,
                          options=["COMPRESS=LZW", "TILED=YES"])
    ae_ds.SetGeoTransform(geotransform)
    ae_ds.SetProjection(ae_srs.ExportToWkt())
    for band_idx in range(4):
        ae_ds.GetRasterBand(band_idx + 1).WriteArray(rgba[:, :, band_idx])
    ae_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    ae_ds = None

    # reproject to EPSG:3857 for Leaflet
    warped_ds = gdal.Warp("", ae_tif, dstSRS="EPSG:3857",
                          resampleAlg=gdal.GRA_NearestNeighbour,
                          format="MEM", dstAlpha=False)

    # write final Cloud-Optimized GeoTIFF
    driver = gdal.GetDriverByName("COG")
    driver.CreateCopy(final_tif, warped_ds,
                      options=["COMPRESS=DEFLATE", "OVERVIEWS=IGNORE_EXISTING"])

    warped_ds = None
    logger.info(f"Wrote {final_tif}")
    os.remove(ae_tif)


def project_data(npz_path: str, radar_lat: float, radar_lon: float,
                 out_dir: str, index: int, is_forecast: bool = False) -> None:
    """Project a detection .npz (inputNF + nfout) into a frame GeoTIFF."""
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

    # render to RGBA then write GeoTIFF
    rgba = _reflectivity_to_rgba(refl, nfout, is_forecast=is_forecast)
    _write_geotiff(rgba, radar_lat, radar_lon, out_dir, index)


def project_forecast(npz_path: str, radar_lat: float, radar_lon: float,
                     out_dir: str, index: int,
                     background_det_npz: str | None = None) -> None:
    """Project a forecast-summary .npz (nfproxy probability grid) into a frame GeoTIFF.

    The background is the frozen radar reflectivity from the most recent
    detection frame (background_det_npz).  Pixels in the nfproxy map at or
    above the 30% threshold are treated as gust-front likelihood and drawn in
    bright purple on top of the reflectivity background, with no dilation.
    """
    data = np.load(npz_path, allow_pickle=True)
    nfproxy = data['nfproxy'].astype(np.float64)
    nfproxy = np.flipud(nfproxy)
    ny, nx = nfproxy.shape

    # start from the frozen reflectivity background of the last detection frame
    if background_det_npz is not None:
        try:
            bg_data = np.load(background_det_npz)
            bg_array = np.flipud(bg_data['inputNF'])
            bg_refl = bg_array[:, :, 1].astype(np.float64)  # channel 1 = reflectivity
            rgba = _reflectivity_to_rgba(bg_refl, nfout=None)
            logger.info(f"Forecast frame {index}: using frozen reflectivity background")
        except Exception as e:
            logger.warning(f"Could not load background reflectivity from {background_det_npz}: {e}")
            rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
    else:
        logger.warning(f"Forecast frame {index}: no background detection npz provided, using transparent background")
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)

    # Draw three contour levels, lowest first so higher-confidence rings overwrite.
    # Each level is skeletonized independently to produce a thin medial-axis line:
    #   30% (orange)          — gust front may pass here
    #   50% (reddish-fuchsia) — moderate confidence
    #   75% (purple)          — high confidence core
    forecast_levels = [
        (30.0, _GF_RGBA_FC_30),
        (50.0, _GF_RGBA_FC_50),
        (75.0, _GF_RGBA_FC_75),
    ]
    any_drawn = False
    for threshold, color in forecast_levels:
        mask = nfproxy >= threshold
        if np.any(mask):
            skel = skeletonize(mask)
            if np.any(skel):
                # disk(1) dilation for visibility — applied after skeletonize so we
                # don't dilate the original blob, only the thin skeleton spine
                dilated = binary_dilation(skel, structure=disk(1))
                rgba[dilated] = color
                logger.info(f"Forecast contour level >={threshold:.0f}%: {np.count_nonzero(skel)} skeleton pixels")
                any_drawn = True
    if not any_drawn:
        logger.info("Forecast frame has no gust-front probability >= 30%")

    _write_geotiff(rgba, radar_lat, radar_lon, out_dir, index)
