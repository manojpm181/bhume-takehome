import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.affinity import translate
from shapely.ops import transform as shp_transform
from affine import Affine
from pyproj import Transformer
import geopandas as gpd
import json, os



_t_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_t_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def to_3857(geom):
    """Reproject shapely geometry from EPSG:4326 → EPSG:3857."""
    return shp_transform(_t_4326_to_3857.transform, geom)


def to_4326(geom):
    """Reproject shapely geometry from EPSG:3857 → EPSG:4326."""
    return shp_transform(_t_3857_to_4326.transform, geom)


def flatten(geom):
    """Flatten MultiPolygon → largest polygon."""
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom




def load_village(village_dir: str) -> dict:
    """Load all files for a village bundle. Plots kept in EPSG:4326."""
    plots = gpd.read_file(os.path.join(village_dir, "input.geojson")).to_crs("EPSG:4326")
    return {
        "plots": plots,
        "imagery_path":    os.path.join(village_dir, "imagery.tif"),
        "boundaries_path": os.path.join(village_dir, "boundaries.tif"),
        "truths_path":     os.path.join(village_dir, "example_truths.geojson"),
    }




def patch_for_plot(geom_4326, raster_path: str, padding_factor: float = 2.5):
    """
    Extract a raster patch around a plot geometry.
    geom_4326: shapely geometry in EPSG:4326
    raster_path: path to a raster in EPSG:3857

    Returns:
        img_array : (H, W, bands) float32, normalised 0–1
        transform : Affine transform for the patch (pixel → EPSG:3857)
        win_bounds_3857 : (minx, miny, maxx, maxy) in EPSG:3857
    """
    geom_4326 = flatten(geom_4326)

    geom_3857 = to_3857(geom_4326)
    bounds = geom_3857.bounds 

    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    hw = (bounds[2] - bounds[0]) * padding_factor / 2
    hh = (bounds[3] - bounds[1]) * padding_factor / 2
    pad = max(hw, hh, 50.0)   

    win_bounds = (cx - pad, cy - pad, cx + pad, cy + pad)

    with rasterio.open(raster_path) as src:
        win = from_bounds(*win_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(window=win, resampling=Resampling.bilinear)
        transform = src.window_transform(win)

    if arr.size == 0:
        raise ValueError(f"Empty patch for geom at {geom_4326.centroid}")

    arr = arr.astype(np.float32)
    for b in range(arr.shape[0]):
        mn, mx = arr[b].min(), arr[b].max()
        if mx > mn:
            arr[b] = (arr[b] - mn) / (mx - mn)

    return np.moveaxis(arr, 0, -1), transform, win_bounds



def lonlat_to_pixel(lon, lat, transform: Affine):
    """EPSG:4326 (lon,lat) → pixel (col, row) in a 3857-transform patch."""
    
    x, y = _t_4326_to_3857.transform(lon, lat)
    inv = ~transform
    col, row = inv * (x, y)
    return int(col), int(row)


def xy3857_to_pixel(x, y, transform: Affine):
    """EPSG:3857 (x,y) → pixel (col, row)."""
    inv = ~transform
    col, row = inv * (x, y)
    return int(col), int(row)


def pixel_to_lonlat(col, row, transform: Affine):
    """Pixel (col,row) → EPSG:4326 (lon,lat). transform is 3857."""
    x, y = transform * (col + 0.5, row + 0.5)
    lon, lat = _t_3857_to_4326.transform(x, y)
    return lon, lat




def geom_centroid(geom):
    c = geom.centroid
    return c.x, c.y


def apply_shift_metres(geom_4326, dx_m: float, dy_m: float):
    """
    Shift a EPSG:4326 geometry by (dx_m, dy_m) metres.
    Works by going through 3857, shifting, converting back.
    """
    geom_4326 = flatten(geom_4326)
    geom_3857 = to_3857(geom_4326)
    shifted_3857 = translate(geom_3857, xoff=dx_m, yoff=dy_m)
    return to_4326(shifted_3857)


def polygon_to_pixel_coords(geom_4326, transform: Affine):
    """
    Convert exterior ring of a EPSG:4326 polygon to pixel coords in a
    3857-transform patch.
    """
    geom_4326 = flatten(geom_4326)
    coords = []
    for lon, lat in geom_4326.exterior.coords:
        col, row = lonlat_to_pixel(lon, lat, transform)
        coords.append((col, row))
    return np.array(coords, dtype=np.int32)




def area_sqm(geom_4326):
    """Area in square metres via EPSG:3857 projection."""
    geom_4326 = flatten(geom_4326)
    geom_3857 = to_3857(geom_4326)
    return abs(geom_3857.area)



def write_predictions(predictions: list, out_path: str):
    """Write predictions list → predictions.geojson in EPSG:4326."""
    features = []
    for p in predictions:
        feat = {
            "type": "Feature",
            "properties": {
                "plot_number": p["plot_number"],
                "status":      p["status"],
                "confidence":  p.get("confidence"),
                "method_note": p.get("method_note", ""),
            },
            "geometry": mapping(p["geometry"]),
        }
        features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f" Written {len(features)} predictions → {out_path}")