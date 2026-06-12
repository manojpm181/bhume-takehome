
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape
import cv2
import rasterio

from utils import (
    patch_for_plot, flatten, to_3857, to_4326,
    apply_shift_metres, polygon_to_pixel_coords, area_sqm,
)


AREA_RATIO_LOW   = 0.55
AREA_RATIO_HIGH  = 1.70
PADDING          = 3.0
MAX_SHIFT_M      = 80.0
XCORR_SCORE_MIN  = 5.0
STRATEGY_B_MIN_SNR = 1.5   




def _render_polygon_mask(geom_4326, transform, shape_hw):
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.float32)
    coords = polygon_to_pixel_coords(geom_4326, transform)
    coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
    if len(coords) < 3:
        return mask
    cv2.polylines(mask, [coords.reshape(-1, 1, 2)],
                  isClosed=True, color=1.0, thickness=2)
    kernel = np.ones((3, 3), np.float32)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask



def _template_match_shift(geom_4326, boundaries_path):
    bnd_arr, bnd_transform, _ = patch_for_plot(geom_4326, boundaries_path, PADDING)
    bnd = gaussian_filter(bnd_arr[:, :, 0].astype(np.float32), sigma=1.5)
    H, W = bnd.shape
    mask = _render_polygon_mask(geom_4326, bnd_transform, (H, W))
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0
    A = np.fft.fft2(bnd)
    B = np.fft.fft2(mask)
    denom = np.abs(A * np.conj(B))
    denom[denom == 0] = 1e-10
    R = (A * np.conj(B)) / denom
    r = np.fft.ifft2(R).real
    peak_idx = np.unravel_index(np.argmax(r), r.shape)
    score = float(r[peak_idx] / (r.mean() + 1e-10))
    dy_px = peak_idx[0] if peak_idx[0] < H // 2 else peak_idx[0] - H
    dx_px = peak_idx[1] if peak_idx[1] < W // 2 else peak_idx[1] - W
    dx_m =  dx_px * abs(bnd_transform.a)
    dy_m = -dy_px * abs(bnd_transform.e)
    if abs(dx_m) > MAX_SHIFT_M or abs(dy_m) > MAX_SHIFT_M:
        return 0.0, 0.0, 0.0
    return dx_m, dy_m, score


def _grid_refine(geom_4326, boundaries_path, init_dx, init_dy,
                 radius_m=8.0, steps=7):
    bnd_arr, bnd_transform, _ = patch_for_plot(geom_4326, boundaries_path, PADDING)
    bnd = gaussian_filter(bnd_arr[:, :, 0].astype(np.float32), sigma=1.5)
    H, W = bnd.shape

    best_score = -1.0
    best_dx, best_dy = init_dx, init_dy
    best_geom = apply_shift_metres(geom_4326, init_dx, init_dy)

    for ddx in np.linspace(-radius_m, radius_m, steps):
        for ddy in np.linspace(-radius_m, radius_m, steps):
            dx = init_dx + ddx
            dy = init_dy + ddy
            cand = apply_shift_metres(geom_4326, dx, dy)
            mask = _render_polygon_mask(cand, bnd_transform, (H, W))
            if mask.sum() < 3:
                continue
            score = float((mask * bnd).sum() / (mask.sum() + 1e-10))
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
                best_geom = cand

    return best_dx, best_dy, best_score, best_geom



def compute_global_median_shift(plots_gdf, imagery_path, boundaries_path,
                                sample_n=120):
    """
    Rasterize all plot edges onto the boundary raster grid,
    phase-correlate to find the systematic offset.
    Returns (dx_m, dy_m) — or (0,0) if signal is too weak.
    """
    print(f"   Rasterizing {len(plots_gdf)} plot edges onto boundary grid...")

    with rasterio.open(boundaries_path) as src:
        bnd_full      = src.read(1).astype(np.float32)
        bnd_transform = src.transform
        bnd_h, bnd_w  = bnd_full.shape

    mx = bnd_full.max()
    if mx > 0:
        bnd_full /= mx

    plot_canvas = np.zeros((bnd_h, bnd_w), dtype=np.float32)
    inv_t = ~bnd_transform

    for _, row in plots_gdf.iterrows():
        geom = flatten(shape(row.geometry))
        geom_3857 = to_3857(geom)
        coords_px = []
        for x, y in geom_3857.exterior.coords:
            col, r_ = inv_t * (x, y)
            coords_px.append((int(col), int(r_)))
        if len(coords_px) < 3:
            continue
        arr = np.array(coords_px, dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, bnd_w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, bnd_h - 1)
        cv2.polylines(plot_canvas, [arr.reshape(-1, 1, 2)],
                      isClosed=True, color=1.0, thickness=1)

    nonzero_pct = 100 * np.count_nonzero(plot_canvas) / plot_canvas.size
    print(f"   Canvas coverage: {nonzero_pct:.2f}%")

    if nonzero_pct < 0.01:
        print("   ⚠️  Canvas empty — returning 0,0")
        return 0.0, 0.0

    bnd_s  = gaussian_filter(bnd_full,    sigma=2.0)
    plot_s = gaussian_filter(plot_canvas, sigma=2.0)

    bnd_d  = bnd_s[::2, ::2]
    plot_d = plot_s[::2, ::2]
    h = min(bnd_d.shape[0], plot_d.shape[0])
    w = min(bnd_d.shape[1], plot_d.shape[1])
    bnd_d  = bnd_d[:h, :w]
    plot_d = plot_d[:h, :w]

    A = np.fft.fft2(bnd_d)
    B = np.fft.fft2(plot_d)
    denom = np.abs(A * np.conj(B))
    denom[denom == 0] = 1e-10
    R = (A * np.conj(B)) / denom
    r = np.fft.ifft2(R).real

    px_w = abs(bnd_transform.a) * 2
    px_h = abs(bnd_transform.e) * 2
    max_px_x = int(MAX_SHIFT_M / px_w) + 1
    max_px_y = int(MAX_SHIFT_M / px_h) + 1

    r_mean = float(np.abs(r).mean())
    r_max  = float(r.max())
    snr    = r_max / (r_mean + 1e-10)
    print(f"   Correlation SNR: {snr:.2f}  (threshold: {STRATEGY_B_MIN_SNR})")

    if snr < STRATEGY_B_MIN_SNR:
        print(f"   ⚠️  SNR too low — boundary signal unreliable for this village")
        print(f"   ⚠️  Returning 0,0 (safer than a wrong shift)")
        return 0.0, 0.0

    
    candidates = []
    for dy_px in range(-max_px_y, max_px_y + 1):
        for dx_px in range(-max_px_x, max_px_x + 1):
            ri = int(dy_px % h)
            ci = int(dx_px % w)
            score = float(r[ri, ci])
            dx_m =  dx_px * px_w
            dy_m = -dy_px * px_h
            candidates.append((dx_m, dy_m, score))

    candidates.sort(key=lambda x: -x[2])

    print(f"   Top peaks within ±{MAX_SHIFT_M:.0f}m:")
    for i, (dx, dy, sc) in enumerate(candidates[:6]):
        print(f"     peak {i+1}: dx={dx:+.1f}m dy={dy:+.1f}m score={sc:.6f}")

    best_dx, best_dy, best_sc = candidates[0]

    
    top     = candidates[:15]
    top_arr = np.array(top)
    w_arr   = top_arr[:, 2] - top_arr[:, 2].min() + 1e-10
    refined_dx = float(np.average(top_arr[:, 0], weights=w_arr))
    refined_dy = float(np.average(top_arr[:, 1], weights=w_arr))

    print(f"   Best:    dx={best_dx:+.1f}m dy={best_dy:+.1f}m")
    print(f"   Refined: dx={refined_dx:+.1f}m dy={refined_dy:+.1f}m")
    return refined_dx, refined_dy




def _area_ratio(geom_4326, recorded_area_sqm):
    drawn = area_sqm(geom_4326)
    if recorded_area_sqm <= 0:
        return 1.0, False
    ratio = drawn / recorded_area_sqm
    return ratio, (ratio < AREA_RATIO_LOW or ratio > AREA_RATIO_HIGH)



def correct_plot(row, global_dx_m, global_dy_m,
                 imagery_path, boundaries_path,
                 is_override=False):

    plot_number = str(row.get("plot_number") or row.get("id") or row.name)
    geom_4326 = flatten(shape(row.geometry))

    recorded_area = 0.0
    for field in ["area", "recorded_area", "shape_area", "Shape_Area"]:
        v = row.get(field)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0:
                    recorded_area = fv
                    break
            except (TypeError, ValueError):
                continue
    # restraint layer
    shift_mag = (
        global_dx_m**2 +
        global_dy_m**2
    ) ** 0.5

    if shift_mag < 2:

        return {
            "plot_number": plot_number,
            "status": "flagged",
            "confidence": None,
            "method_note": "skip_small_shift",
            "geometry": geom_4326,
        }
    shifted = apply_shift_metres(
        geom_4326,
        global_dx_m,
        global_dy_m
    )

    if is_override:

        snapped = shifted

        best_dx = global_dx_m
        best_dy = global_dy_m

        method_tag = "exact_truth_override"

        dx_local = 0.0
        dy_local = 0.0

        try:

            bnd_arr, bnd_t, _ = patch_for_plot(
                snapped,
                boundaries_path,
                PADDING
            )

            bnd_sig = gaussian_filter(
                bnd_arr[:, :, 0].astype(np.float32),
                sigma=1.5
            )

            H, W = bnd_sig.shape

            snap_mask = _render_polygon_mask(
                snapped,
                bnd_t,
                (H, W)
            )

            snap_score = float(
                (snap_mask * bnd_sig).sum()
                / (snap_mask.sum() + 1e-10)
            )

        except Exception:
            snap_score = 0.0

        try:
            _, _, tmatch_score = _template_match_shift(
                snapped,
                boundaries_path
            )
        except Exception:
            tmatch_score = 0.0

    else:

        try:

            dx_local, dy_local, tmatch_score = _template_match_shift(
                shifted,
                boundaries_path
            )

            if (
                tmatch_score <= XCORR_SCORE_MIN
                or abs(dx_local) > MAX_SHIFT_M
                or abs(dy_local) > MAX_SHIFT_M
            ):
                dx_local = 0.0
                dy_local = 0.0
                tmatch_score = 0.0

        except Exception:

            dx_local = 0.0
            dy_local = 0.0
            tmatch_score = 0.0

        try:

            bnd_arr, bnd_t, _ = patch_for_plot(
                shifted,
                boundaries_path,
                PADDING
            )

            bnd_base = gaussian_filter(
                bnd_arr[:, :, 0].astype(np.float32),
                sigma=1.5
            )

            H, W = bnd_base.shape

            base_mask = _render_polygon_mask(
                shifted,
                bnd_t,
                (H, W)
            )

            snap_score = float(
                (base_mask * bnd_base).sum()
                / (base_mask.sum() + 1e-10)
            )

        except Exception:
            snap_score = 0.0

        after_local = (
            apply_shift_metres(
                shifted,
                dx_local,
                dy_local
            )
            if tmatch_score > XCORR_SCORE_MIN
            else shifted
        )

        if tmatch_score <= XCORR_SCORE_MIN:
            dx_local = 0.0
            dy_local = 0.0

        try:

            best_dx, best_dy, grid_score, snapped = _grid_refine(
                geom_4326,
                boundaries_path,
                global_dx_m + dx_local,
                global_dy_m + dy_local,
                radius_m=5.0,
                steps=5
            )

            if grid_score <= snap_score * 1.05:

                snapped = after_local

                best_dx = global_dx_m + dx_local
                best_dy = global_dy_m + dy_local

            else:
                snap_score = grid_score

        except Exception:

            snapped = after_local

            best_dx = global_dx_m + dx_local
            best_dy = global_dy_m + dy_local

        method_tag = "grid_refined"

    ratio, area_flagged = _area_ratio(
        snapped,
        recorded_area
    )

    

    total_m = (best_dx**2 + best_dy**2) ** 0.5

    
    shift_conf = float(
        np.clip(
            1.0 - (total_m / MAX_SHIFT_M) ** 0.5,
            0.0,
            1.0
        )
    )

    
    snap_conf = float(
        np.clip(
            snap_score / 0.25,
            0.0,
            1.0
        )
    )

    
    area_conf = float(
        np.clip(
            1.0 - abs(ratio - 1.0) / 0.5,
            0.0,
            1.0
        )
    )

    
    local_m = (dx_local**2 + dy_local**2) ** 0.5

    consistency_conf = float(
        np.clip(
            1.0 - local_m / 30.0,
            0.0,
            1.0
        )
    )

    
    conf = (
        0.25 * shift_conf +
        0.45 * snap_conf +
        0.20 * area_conf +
        0.10 * consistency_conf
    )
    conf = conf ** 1.5

    conf = float(
        np.clip(
            conf,
            0.05,
            0.95
        )
    )

    if area_flagged:
        conf *= 0.30



    if area_flagged or conf < 0.28:

        status = "flagged"
        out_geom = geom_4326
        conf = None

    else:

        status = "corrected"
        out_geom = snapped
        out_geom = flatten(out_geom)

        try:

            if not out_geom.is_valid:
                out_geom = out_geom.buffer(0)

            if out_geom.is_empty:
                out_geom = geom_4326
                status = "flagged"

        except Exception:

            out_geom = geom_4326
            status = "flagged"


    method = (
        f"{method_tag} "
        f"global=({global_dx_m:.1f},{global_dy_m:.1f}m) "
        f"snap={snap_score:.3f} "
        f"tmatch={tmatch_score:.2f} "
        f"total=({best_dx:.1f},{best_dy:.1f}m) "
        f"area_ratio={ratio:.2f} "
        f"flagged={area_flagged}"
    )

    return {
        "plot_number": plot_number,
        "status": status,
        "confidence": conf,
        "method_note": method,
        "geometry": out_geom,
    }
