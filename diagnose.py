
import sys, os, json
import numpy as np

print("=== DIAGNOSTIC: Why is global shift 0? ===\n")


try:
    import rasterio
    print("✅ rasterio imported OK")
except ImportError as e:
    print(f"❌ rasterio import failed: {e}")
    sys.exit(1)


for village in ["vadnerbhairav", "malatavadi"]:
    print(f"\n--- {village} ---")
    for fname in ["imagery.tif", "boundaries.tif"]:
        path = f"data/{village}/{fname}"
        if not os.path.exists(path):
            print(f"  ❌ MISSING: {path}")
            continue
        try:
            with rasterio.open(path) as src:
                print(f"  ✅ {fname}: CRS={src.crs}  shape={src.shape}  bands={src.count}  dtype={src.dtypes[0]}")
                print(f"     bounds={src.bounds}")
        except Exception as e:
            print(f"  ❌ {fname} open failed: {e}")


print("\n--- Patch extraction test ---")
try:
    import geopandas as gpd
    from shapely.geometry import shape
    from scipy.ndimage import sobel

    village = "vadnerbhairav"
    plots = gpd.read_file(f"data/{village}/input.geojson").to_crs("EPSG:4326")
    row = plots.iloc[0]
    geom = shape(row.geometry)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    print(f"  Plot 0 geometry type: {geom.geom_type}")
    print(f"  Plot 0 bounds: {geom.bounds}")

    imagery_path    = f"data/{village}/imagery.tif"
    boundaries_path = f"data/{village}/boundaries.tif"

    with rasterio.open(imagery_path) as src:
        print(f"  Imagery CRS: {src.crs}")
        rb = src.bounds
        gb = geom.bounds
        overlap = (gb[0] < rb.right and gb[2] > rb.left and
                   gb[1] < rb.top  and gb[3] > rb.bottom)
        print(f"  Plot bbox overlaps imagery: {overlap}")
        if not overlap:
            print(f"  ⚠️  PLOT: {gb}")
            print(f"  ⚠️  RASTER: {rb}")

    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    bounds = geom.bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    pad = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) * 2.5 / 2
    win_bounds = (cx-pad, cy-pad, cx+pad, cy+pad)
    print(f"  Window bounds: {win_bounds}")

    with rasterio.open(imagery_path) as src:
        win = from_bounds(*win_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(window=win, resampling=Resampling.bilinear)
        print(f"  Imagery patch shape: {arr.shape}  min={arr.min():.1f} max={arr.max():.1f}")

    with rasterio.open(boundaries_path) as src:
        win = from_bounds(*win_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr2 = src.read(window=win, resampling=Resampling.bilinear)
        print(f"  Boundary patch shape: {arr2.shape}  min={arr2.min():.4f} max={arr2.max():.4f}")

    print(f"  Boundary nonzero pixels: {np.count_nonzero(arr2)} / {arr2.size}")
    print(f"  Boundary mean: {arr2.mean():.4f}  std: {arr2.std():.4f}")

    gray = arr.astype(np.float32)
    if gray.shape[0] >= 3:
        gray = 0.299*gray[0] + 0.587*gray[1] + 0.114*gray[2]
    else:
        gray = gray[0]
    mx = gray.max()
    if mx > 0: gray /= mx

    bnd = arr2.astype(np.float32)[0]
    mx2 = bnd.max()
    if mx2 > 0: bnd /= mx2

    sx = sobel(gray, axis=1);   sy = sobel(gray, axis=0)
    img_edges = np.hypot(sx, sy)
    sx2 = sobel(bnd, axis=1);   sy2 = sobel(bnd, axis=0)
    bnd_edges = np.hypot(sx2, sy2)

    print(f"  Image edge map:    min={img_edges.min():.4f} max={img_edges.max():.4f} mean={img_edges.mean():.4f}")
    print(f"  Boundary edge map: min={bnd_edges.min():.4f} max={bnd_edges.max():.4f} mean={bnd_edges.mean():.4f}")

    h = min(img_edges.shape[0], bnd_edges.shape[0])
    w = min(img_edges.shape[1], bnd_edges.shape[1])
    a_e = img_edges[:h,:w];  b_e = bnd_edges[:h,:w]

    A = np.fft.fft2(a_e);  B = np.fft.fft2(b_e)
    denom = np.abs(A * np.conj(B))
    denom[denom==0] = 1e-10
    R = (A * np.conj(B)) / denom
    r = np.fft.ifft2(R).real
    peak_idx = np.unravel_index(np.argmax(r), r.shape)
    score = r[peak_idx] / (r.mean() + 1e-10)
    print(f"  Phase correlation peak: {peak_idx}  score={score:.3f}")
    print(f"  Score > 0.05? {score > 0.05}  ← must be True to register a shift")

    with rasterio.open(imagery_path) as src:
        pixel_m = abs(src.transform.a) * 111320
        print(f"  Pixel size: {pixel_m:.2f} metres/pixel")

    # Also check what fields are in the geojson
    print(f"\n  Plot columns: {list(plots.columns)}")
    print(f"  First row values: {dict(plots.iloc[0].drop('geometry'))}")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()

print("\n=== END DIAGNOSTIC ===")