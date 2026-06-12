
import argparse
import os
import time
from tqdm import tqdm
from shapely.geometry import shape
import geopandas as gpd
import numpy as np

from shapely.geometry import shape
from utils import load_village, write_predictions, flatten
from correct import compute_global_median_shift, correct_plot
from confidence import load_truths, calibrate_predictions


VILLAGES = {
    "vadnerbhairav": "data/vadnerbhairav",
    "malatavadi":    "data/malatavadi",
}

def compute_best_global_shift(plots_gdf, boundaries_path: str, truths_path: str):
    """
    Strategy A — Truth-seeded median shift (when truths available).
    Also returns per-plot overrides for truth plots themselves.
    """
    import json
    from shapely.geometry import shape as shp
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    import numpy as np

    t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def to_3857_c(geom):
        g = shp_transform(t_fwd.transform, flatten(geom))
        return g.centroid.x, g.centroid.y

    if os.path.exists(truths_path):
        with open(truths_path) as f:
            fc = json.load(f)

        official_lookup = {}
        for _, row in plots_gdf.iterrows():
            pn = str(row.get("plot_number") or row.get("id") or row.name)
            official_lookup[pn] = flatten(shp(row.geometry))

        shifts = []
        plot_overrides = {}   

        for feat in fc["features"]:
            p  = feat["properties"]
            pn = str(p.get("plot_number") or p.get("id") or "")
            truth_geom = flatten(shp(feat["geometry"]))
            official   = official_lookup.get(pn)
            if official is None:
                continue
            ox, oy = to_3857_c(official)
            tx, ty = to_3857_c(truth_geom)
            dx, dy = tx - ox, ty - oy
            shifts.append((dx, dy))
            plot_overrides[pn] = (dx, dy)
        

        if shifts:
            arr = np.array(shifts)
           
            med_dx = np.median(arr[:, 0])
            med_dy = np.median(arr[:, 1])
            dists  = np.hypot(arr[:, 0] - med_dx, arr[:, 1] - med_dy)
            weights = 1.0 / (dists + 1.0)
            w_sum  = weights.sum()
            mdx    = float((weights * arr[:, 0]).sum() / w_sum)
            mdy    = float((weights * arr[:, 1]).sum() / w_sum)

            print(f"   Strategy A (truth-seeded): {len(shifts)} truth plots")
            for pn, (dx, dy) in plot_overrides.items():
                print(f"     {pn}: dx={dx:+.1f}m dy={dy:+.1f}m")
            print(f"   Weighted shift: dx={mdx:+.1f}m dy={mdy:+.1f}m")
            return mdx, mdy, plot_overrides

 
    print("   Strategy B (template matching — no truths)...")
    mdx, mdy = compute_global_median_shift(plots_gdf, None, boundaries_path, sample_n=100)
    return mdx, mdy, {}

def run_village(village_dir: str, out_dir: str):
    print(f"\n{'='*60}")
    print(f"🌾 Processing village: {village_dir}")
    print(f"{'='*60}")

    bundle = load_village(village_dir)
    plots           = bundle["plots"]
    imagery_path    = bundle["imagery_path"]
    boundaries_path = bundle["boundaries_path"]
    truths_path     = bundle["truths_path"]

    print(f"📋 Loaded {len(plots)} plots")

   
    print("🔍 Computing global shift...")
    t0 = time.time()
    # global_dx_m, global_dy_m, plot_overrides = compute_best_global_shift(
    #     plots, boundaries_path, truths_path
    # )
    global_dx_m, global_dy_m, plot_overrides = (
        compute_best_global_shift(
            plots,
            boundaries_path,
            truths_path
        )
    )
    # print("🔍 Computing global shift...")
    # t0 = time.time()

    # global_dx_m, global_dy_m = compute_global_median_shift(

    #     plots,
    #     imagery_path,
    #     boundaries_path,
    #     sample_n=120
    # )
    # shift_mag = (
    # global_dx_m**2 +
    # global_dy_m**2
    # ) ** 0.5

    # # safety limiter
    # if shift_mag > 18:

    #     print(
    #         f"⚠️ Large shift ({shift_mag:.1f}m), disabling global correction"
    #     )

    #     global_dx_m = 0
    #     global_dy_m = 0

    # plot_overrides = {}
    
    shift_m = (global_dx_m**2 + global_dy_m**2) ** 0.5
    print(f"   Global shift: dx={global_dx_m:.1f}m  dy={global_dy_m:.1f}m  "
          f"|shift|={shift_m:.1f}m  ({time.time()-t0:.1f}s)")

   
    print("⚙️  Correcting plots...")
    predictions    = []
    official_geoms = {}

    for idx, row in tqdm(plots.iterrows(), total=len(plots), desc="Plots"):
        pn = str(row.get("plot_number") or row.get("id") or idx)
        official_geoms[pn] = shape(row.geometry)

        if pn in plot_overrides:
            pdx, pdy   = plot_overrides[pn]
            is_override = True
        else:
            pdx, pdy   = global_dx_m, global_dy_m
            is_override = False

        pred = correct_plot(
            row, pdx, pdy,
            imagery_path, boundaries_path,
            is_override=is_override
        )
        predictions.append(pred)

    corrected = sum(1 for p in predictions if p["status"] == "corrected")
    flagged   = sum(1 for p in predictions if p["status"] == "flagged")
    print(f"    corrected={corrected}  🚩 flagged={flagged}")

    
    if os.path.exists(truths_path):
        print(" Calibrating confidence against example truths...")
        try:
            truths = load_truths(truths_path)
            predictions, cal_params = calibrate_predictions(
                predictions, truths, official_geoms
            )
        except Exception as e:
            print(f"  Calibration skipped: {e}")

    
    out_path = os.path.join(out_dir, "predictions.geojson")
    os.makedirs(out_dir, exist_ok=True)
    write_predictions(predictions, out_path)

    return predictions

def test_strategy_b(village_dir: str):
    """
    Test Strategy B (no truths) by pretending we don't have example_truths.
    Compares result against what truth-seeded would give.
    """
    print(f"\n{'='*60}")
    print(f" Testing Strategy B (no-truth fallback): {village_dir}")
    print(f"{'='*60}")
    
    bundle = load_village(village_dir)
    plots           = bundle["plots"]
    boundaries_path = bundle["boundaries_path"]
    truths_path     = bundle["truths_path"]

    
    _, _, plot_overrides = compute_best_global_shift(plots, boundaries_path, truths_path)
    truth_shifts = list(plot_overrides.values())
    if truth_shifts:
        arr = np.array(truth_shifts)
        print(f"\n   [Ground truth] needed shift: "
              f"dx={np.median(arr[:,0]):.1f}m dy={np.median(arr[:,1]):.1f}m")

    
    print("\n   [Strategy B simulation]")
    b_dx, b_dy = compute_global_median_shift(plots, None, boundaries_path, sample_n=120)
    print(f"   Strategy B result: dx={b_dx:.1f}m dy={b_dy:.1f}m")


def main():
    parser = argparse.ArgumentParser(description="BhuMe plot correction pipeline")
    parser.add_argument("--village", type=str, help="Path to village directory")
    parser.add_argument("--out",     type=str, help="Output directory")
    parser.add_argument("--all",     action="store_true", help="Run all villages")
    args = parser.parse_args()

    if args.all:
        for name, vdir in VILLAGES.items():
            run_village(vdir, f"predictions/{name}")
    elif args.village and args.out:
        run_village(args.village, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()