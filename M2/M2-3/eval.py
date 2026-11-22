
from ultralytics import YOLO
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent

model = YOLO(str(BASE / "models/best.pt"))

model.val(
    data=str(BASE / "dataset/final_v2/data.yaml"),
    split="test",
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=True,
    save_txt=True,
    save_conf=True,
    project=str(BASE / "runs/detect"),
    name="final_eval"
)
