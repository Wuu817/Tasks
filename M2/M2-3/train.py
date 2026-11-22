
from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parent

def main():

    # 使用第一版训练好的模型作为初始权重
    model = YOLO(str(BASE / "archive/models/best_v1.pt"))

    # 继续训练第二版模型
    model.train(
        data=str(BASE / "dataset/final_v2/data.yaml"),

        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,
        patience=20,

        project=str(BASE / "runs/detect"),
        name="final_v2",

        seed=42,
        plots=True
    )

if __name__ == "__main__":
    main()
