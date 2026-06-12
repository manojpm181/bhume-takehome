
import json, os
import numpy as np
from shapely.geometry import shape
from pyproj import Transformer

t_fwd = Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
t_inv = Transformer.from_crs("EPSG:3857","EPSG:4326",always_xy=True)

def to_3857(geom):
    from shapely.ops import transform
    return transform(t_fwd.transform, geom)

def centroid_3857(geom):
    g = to_3857(geom)
    return g.centroid.x, g.centroid.y

def load_fc(path):
    with open(path) as f: fc = json.load(f)
    out = {}
    for feat in fc["features"]:
        p = feat["properties"]
        pn = str(p.get("plot_number") or p.get("id") or "")
        out[pn] = shape(feat["geometry"])
    return out

for village in ["vadnerbhairav","malatavadi"]:
    truth_path = f"data/{village}/example_truths.geojson"
    input_path = f"data/{village}/input.geojson"
    pred_path  = f"predictions/{village}/predictions.geojson"
    if not os.path.exists(truth_path):
        continue

    truths  = load_fc(truth_path)
    inputs  = load_fc(input_path)
    preds   = load_fc(pred_path) if os.path.exists(pred_path) else {}

    print(f"\n{'='*65}")
    print(f"Village: {village}  ({len(truths)} truth plots)")
    print(f"{'='*65}")
    print(f"{'Plot':<8} {'Need dx':>10} {'Need dy':>10} {'|drift|':>9} {'Our dx':>9} {'Our dy':>9}")
    print("-"*65)

    true_shifts = []
    for pn, truth in truths.items():
        official = inputs.get(pn)
        if official is None:
            print(f"{pn:<8} missing from input")
            continue

        ox, oy = centroid_3857(official)
        tx, ty = centroid_3857(truth)
        need_dx = tx - ox
        need_dy = ty - oy
        true_shifts.append((need_dx, need_dy))

        pred = preds.get(pn)
        if pred:
            px, py = centroid_3857(pred)
            our_dx = px - ox
            our_dy = py - oy
        else:
            our_dx, our_dy = 0, 0

        drift = (need_dx**2 + need_dy**2)**0.5
        print(f"{pn:<8} {need_dx:>+10.1f} {need_dy:>+10.1f} {drift:>9.1f}m {our_dx:>+9.1f} {our_dy:>+9.1f}")

    if true_shifts:
        arr = np.array(true_shifts)
        print("-"*65)
        print(f"TRUE needed  median: dx={np.median(arr[:,0]):+.1f}m  dy={np.median(arr[:,1]):+.1f}m")
        print(f"TRUE needed  mean:   dx={np.mean(arr[:,0]):+.1f}m  dy={np.mean(arr[:,1]):+.1f}m")
        print(f"TRUE needed  std:    dx={np.std(arr[:,0]):.1f}m   dy={np.std(arr[:,1]):.1f}m")