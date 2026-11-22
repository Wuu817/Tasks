import cv2
import csv
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from detect import detect

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

VIDEOS = ["day_01", "dark_01", "backlight_01"]
STATES = ["red", "yellow", "green", "none"]
rows = []

def predict(frame):
    annotated, lines = detect(frame)
    for line in lines:
        match = re.search(
            r"(RED|YELLOW|GREEN)\s+x=([\d.]+)\s+y=([\d.]+).*?radius=([\d.]+)",
            str(line), re.I
        )
        if match:
            return (
                match.group(1).lower(),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
                annotated
            )
    return "none", None, None, None, annotated

def metrics(data):
    lights = [r for r in data if r["gt"] != "none"]
    empty = [r for r in data if r["gt"] == "none"]
    errors = [r["error"] for r in data if r["error"] is not None]
    normalized = [r["normalized_error"] for r in data
                  if r["normalized_error"] is not None]

    def rate(a, b):
        return f"{100*a/b:.2f}%" if b else "N/A"

    return {
        "frames": len(data),
        "accuracy": rate(sum(r["gt"] == r["pred"] for r in lights), len(lights)),
        "miss": rate(sum(r["pred"] == "none" for r in lights), len(lights)),
        "false_positive": rate(sum(r["pred"] != "none" for r in empty), len(empty)),
        "P50": f"{np.percentile(errors, 50):.2f}" if errors else "N/A",
        "P95": f"{np.percentile(errors, 95):.2f}" if errors else "N/A",
        "normalized_P50": f"{np.percentile(normalized, 50):.3f}" if normalized else "N/A",
        "normalized_P95": f"{np.percentile(normalized, 95):.3f}" if normalized else "N/A"
    }

for name in VIDEOS:
    label_path = ROOT / "data/labels" / f"{name}.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))["frames"]

    cap = cv2.VideoCapture(str(ROOT / "data/raw" / f"{name}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"视频打不开：{name}")

    saved = 0
    for i, item in enumerate(labels):
        frame_id = int(item["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{name} 第{frame_id}帧读取失败")

        pred, x, y, radius, annotated = predict(frame)
        gt = item["state"].lower()
        center = item.get("center")
        gt_radius = item.get("radius_px")

        error = None
        normalized = None

        if gt != "none" and pred != "none" and center is not None:
            error = float(np.hypot(x - center[0], y - center[1]))
            if gt_radius and gt_radius > 0:
                normalized = error / gt_radius

        rows.append({
            "video": name,
            "split": "test" if name == "backlight_01" else "tune",
            "frame": frame_id,
            "time": item["t"],
            "gt": gt,
            "pred": pred,
            "gt_center": str(center),
            "pred_x": x,
            "pred_y": y,
            "pred_radius": radius,
            "error": error,
            "normalized_error": normalized
        })

        if i in (0, 50):
            folder = RESULTS / name
            folder.mkdir(exist_ok=True)
            image = annotated if annotated is not None else frame
            cv2.imwrite(str(folder / f"annotated_{i+1:03d}.png"), image)
            saved += 1

    cap.release()
    print(f"完成 {name}: {len(labels)} 帧，保存 {saved} 张检测图")

fields = list(rows[0].keys())
with (RESULTS / "eval.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with (RESULTS / "confusion_matrix.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["GT / PRED"] + STATES)
    for gt in STATES:
        writer.writerow([gt] + [
            sum(r["gt"] == gt and r["pred"] == pred for r in rows)
            for pred in STATES
        ])

summary = []
for name in VIDEOS:
    m = metrics([r for r in rows if r["video"] == name])
    summary.append(f"\n{name} ({'测试段' if name == 'backlight_01' else '调参段'})")
    summary.extend(f"{k}: {v}" for k, v in m.items())

summary.append("\n全部视频")
summary.extend(f"{k}: {v}" for k, v in metrics(rows).items())

text = "\n".join(summary)
(RESULTS / "eval_summary.txt").write_text(text, encoding="utf-8")
print(text)
print("\n评估完成：results/eval.csv")
