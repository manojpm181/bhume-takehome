
import json, argparse, os
import numpy as np
from shapely.geometry import shape


def iou(a, b):
    try:
        return a.intersection(b).area / a.union(b).area
    except:
        return 0.0


def centroid_dist_m(a, b):
    dx = (a.centroid.x - b.centroid.x) * 111320
    dy = (a.centroid.y - b.centroid.y) * 111320
    return (dx**2 + dy**2) ** 0.5


def load_fc(path):
    with open(path) as f:
        fc = json.load(f)
    return {
        str(feat["properties"].get("plot_number") or feat["properties"].get("id")):
        shape(feat["geometry"])
        for feat in fc["features"]
    }


def score(village: str):
    pred_path   = f"predictions/{village}/predictions.geojson"
    truth_path  = f"data/{village}/example_truths.geojson"
    input_path  = f"data/{village}/input.geojson"

    if not os.path.exists(pred_path):
        print(f"❌ No predictions found at {pred_path}")
        return

    preds  = load_fc(pred_path)
    truths = load_fc(truth_path)
    inputs = load_fc(input_path)

    print(f"\n{'='*55}")
    print(f"🏆 Scoring: {village}")
    print(f"{'='*55}")
    print(f"{'Plot':<12} {'Official IoU':>12} {'Pred IoU':>10} {'Δ IoU':>8} {'Dist(m)':>9}")
    print("-"*55)

    iou_gains, pred_ious, confidences, labels = [], [], [], []

    with open(pred_path) as f:
        pred_fc = json.load(f)
    conf_map = {
        str(f["properties"].get("plot_number") or f["properties"].get("id")):
        f["properties"].get("confidence")
        for f in pred_fc["features"]
    }

    for pn, truth in truths.items():
        official = inputs.get(pn)
        pred     = preds.get(pn)
        if official is None or pred is None:
            continue

        off_iou  = iou(official, truth)
        pred_iou = iou(pred,     truth)
        dist     = centroid_dist_m(pred, truth)
        delta    = pred_iou - off_iou

        iou_gains.append(delta)
        pred_ious.append(pred_iou)

        conf = conf_map.get(pn)
        if conf is not None:
            confidences.append(conf)
            labels.append(1 if pred_iou >= 0.5 else 0)

        symbol = "✅" if pred_iou >= 0.5 else ("⬆️" if delta > 0 else "⬇️")
        print(f"{pn:<12} {off_iou:>12.3f} {pred_iou:>10.3f} "
              f"{delta:>+8.3f} {dist:>8.1f}m  {symbol}")

    print("-"*55)
    if iou_gains:
        print(f"Mean IoU gain : {np.mean(iou_gains):+.3f}")
        print(f"Mean pred IoU : {np.mean(pred_ious):.3f}")
        hits = sum(1 for x in pred_ious if x >= 0.5)
        print(f"Hits (IoU≥0.5): {hits}/{len(pred_ious)}")

    # AUC (confidence calibration score)
    if len(confidences) >= 2 and len(set(labels)) == 2:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels, confidences)
            print(f"Confidence AUC: {auc:.3f}  {'⭐ GREAT' if auc>0.7 else '(needs work)'}")
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--village", required=True)
    args = parser.parse_args()
    score(args.village)