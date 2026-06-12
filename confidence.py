
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import shape
import geopandas as gpd
import json




def iou(geom_a, geom_b) -> float:
    try:
        inter = geom_a.intersection(geom_b).area
        union = geom_a.union(geom_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0



def load_truths(truths_path: str) -> dict:
    """Returns {plot_number: shapely_geometry}."""
    with open(truths_path) as f:
        fc = json.load(f)
    out = {}
    for feat in fc["features"]:
        pn = feat["properties"].get("plot_number") or feat["properties"].get("id")
        out[str(pn)] = shape(feat["geometry"])
    return out


def _sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


def _nll(params, scores, labels):
    a, b = params
    p = _sigmoid(np.array(scores), a, b)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    labels = np.array(labels)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def fit_calibration(raw_scores: list, labels: list):
    """
    Fit Platt scaling parameters.
    labels: 1 if IoU >= 0.5, else 0.
    Returns (a, b).
    """
    if len(raw_scores) < 3:
        return 1.0, 0.0

    result = minimize(_nll, [1.0, 0.0], args=(raw_scores, labels),
                      method="Nelder-Mead", options={"maxiter": 1000})
    return result.x[0], result.x[1]


def calibrate_predictions(predictions: list, truths: dict,
                           official_geoms: dict) -> tuple:
    """
    Calibrate confidences. If all truth labels are identical (all correct
    or all wrong), Platt scaling collapses — skip it and return as-is.
    The real calibration signal comes from the hidden set, not 6 examples.
    """
    raw_scores, labels = [], []

    for pred in predictions:
        pn = str(pred["plot_number"])
        if pn not in truths or pred["status"] != "corrected":
            continue
        truth_geom = truths[pn]
        pred_geom  = pred["geometry"]
        pred_iou   = iou(pred_geom, truth_geom)
        raw_scores.append(pred["confidence"])
        labels.append(1.0 if pred_iou >= 0.5 else 0.0)

    if len(raw_scores) < 2:
        print("   Calibration skipped: too few truth samples")
        return predictions, (1.0, 0.0)

    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print(f"   Calibration skipped: all labels={list(unique_labels)} "
              f"(need both 0 and 1 to fit Platt scaling)")
        print(f"   Raw confidence range: "
              f"[{min(raw_scores):.3f}, {max(raw_scores):.3f}]")
        return predictions, (1.0, 0.0)

    a, b = fit_calibration(raw_scores, labels)
    calibrated = []
    for pred in predictions:
        p = dict(pred)
        if p["status"] == "corrected" and p["confidence"] is not None:
            p["confidence"] = float(_sigmoid(p["confidence"], a, b))
        calibrated.append(p)

    print(f"   Calibration: a={a:.3f} b={b:.3f} on {len(raw_scores)} samples")
    return calibrated, (a, b)