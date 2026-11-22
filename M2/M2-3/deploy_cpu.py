
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time

from pathlib import Path
from collections import deque

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models/best.onnx"

NAMES = [
    "coca_cola",
    "pepsi",
    "sprite",
    "cup",
    "tissue",
    "ointment"
]

COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0)
]

IMG_SIZE = 640
CONF = 0.25
IOU = 0.45

SAVE_DIR = BASE / "results/deploy_cpu"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class CPUDetector:

    def __init__(self):

        # 设置CPU推理线程
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1

        # 加载ONNX模型
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        print("模型加载成功")
        print("运行设备:", self.session.get_providers())


    # ==================== 预处理 ====================

    def preprocess(self, image):

        h, w = image.shape[:2]

        # 等比例缩放
        ratio = min(IMG_SIZE / w, IMG_SIZE / h)

        new_w = round(w * ratio)
        new_h = round(h * ratio)

        resized = cv2.resize(image,(new_w, new_h),interpolation=cv2.INTER_LINEAR)

        # 计算填充量
        dw = IMG_SIZE - new_w
        dh = IMG_SIZE - new_h

        left = round(dw / 2 - 0.1)
        right = round(dw / 2 + 0.1)

        top = round(dh / 2 - 0.1)
        bottom = round(dh / 2 + 0.1)

        # Letterbox：填充灰色边框
        padded = cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114, 114, 114))

        # BGR转RGB
        rgb = cv2.cvtColor(padded,cv2.COLOR_BGR2RGB)

        # HWC转CHW，并归一化
        tensor = rgb.transpose(2, 0, 1)
        tensor = tensor.astype(np.float32) / 255.0

        # 增加batch维度
        tensor = np.expand_dims(tensor, axis=0)

        tensor = np.ascontiguousarray(tensor)

        return tensor, ratio, left, top


    # ==================== 后处理 ====================

    def postprocess(self, output, image_shape,ratio, left, top):

        h, w = image_shape[:2]

        # (1, 10, 8400) -> (8400, 10)
        predictions = output[0].T

        # 前4列是框坐标，后6列是类别分数
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # 过滤低置信度候选框
        mask = confidences >= CONF

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # xywh中心坐标 -> 左上角xywh
        nms_boxes = boxes.copy()

        nms_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        nms_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

        keep = []

        # 按类别分别执行NMS
        for cls in np.unique(class_ids):

            indices = np.where(class_ids == cls)[0]

            selected = cv2.dnn.NMSBoxes(nms_boxes[indices].tolist(),confidences[indices].tolist(),CONF,IOU)

            if len(selected) > 0:

                selected = np.array(selected).reshape(-1)

                keep.extend(indices[selected].tolist())

        # 按置信度排序
        keep.sort(key=lambda i: confidences[i],reverse=True)

        detections = []

        for i in keep:

            cx, cy, bw, bh = boxes[i]

            # 转换成xyxy坐标
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # 从640×640映射回原图坐标
            x1 = (x1 - left) / ratio
            y1 = (y1 - top) / ratio
            x2 = (x2 - left) / ratio
            y2 = (y2 - top) / ratio

            # 限制在原图范围内
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append([x1, y1, x2, y2,confidences[i],class_ids[i]])

        return np.array(detections,dtype=np.float32).reshape(-1, 6)


    # ==================== 完整检测流程 ====================

    def detect(self, image):

        t0 = time.perf_counter()

        # 1. 预处理
        tensor, ratio, left, top = self.preprocess(image)

        t1 = time.perf_counter()

        # 2. ONNX CPU推理
        output = self.session.run(None,{self.input_name: tensor})[0]

        t2 = time.perf_counter()

        # 3. 后处理
        detections = self.postprocess(output,image.shape,ratio,left,top)

        t3 = time.perf_counter()

        # 各阶段耗时，单位毫秒
        timings = {
            "preprocess": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000
        }

        return detections, timings


# ==================== 绘制检测结果 ====================

def draw(image, detections, fps=None):

    for box in detections:

        x1, y1, x2, y2, conf, cls = box

        cls = int(cls)

        color = COLORS[cls]

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # 绘制检测框
        cv2.rectangle(image,(x1, y1),(x2, y2),color,2)

        text = f"{NAMES[cls]} {conf:.2f}"

        cv2.putText(image,text,(x1, max(y1 - 8, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # 显示FPS
    if fps is not None:

        cv2.putText(image,f"CPU FPS: {fps:.1f}",(15, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    return image


# ==================== 图片模式 ====================

def run_image(detector, path):

    image = cv2.imread(str(path))

    if image is None:
        print("无法读取图片:", path)
        return

    detections, timings = detector.detect(image)

    result = draw(image.copy(), detections)

    print("检测目标数量:", len(detections))

    for box in detections:

        cls = int(box[5])
        conf = box[4]

        print(f"{NAMES[cls]}: {conf:.3f}")

    print("\n各阶段耗时:")

    for name, value in timings.items():
        print(f"{name}: {value:.2f} ms")

    # 保存检测图片
    save_path = SAVE_DIR / "image_result.jpg"

    cv2.imwrite(str(save_path), result)

    print("结果已保存:", save_path)

    cv2.imshow("ONNX CPU Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================== 视频与摄像头模式 ====================

def run_stream(detector, source):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("无法打开视频或摄像头")
        return

    # 摄像头分辨率
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 最近30帧耗时，用于计算FPS
    frame_times = deque(maxlen=30)

    print("按 S 保存当前画面")
    print("按 Q 退出")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        detections, timings = detector.detect(frame)

        # 使用此前帧的端到端平均耗时计算FPS
        fps = 0.0

        if frame_times:
            fps = len(frame_times) / sum(frame_times)

        result = draw(frame.copy(),detections,fps)

        cv2.imshow("ONNX CPU Detection", result)

        key = cv2.waitKey(1) & 0xFF

        # 记录当前帧完整处理耗时
        frame_times.append(time.perf_counter() - t0)

        if key == ord("q"):
            break

        elif key == ord("s"):
            number = 1
            save_path = SAVE_DIR / f"frame_{number:03d}.jpg"
            while save_path.exists():
                number += 1
                save_path = SAVE_DIR / f"frame_{number:03d}.jpg"
            cv2.imwrite(str(save_path), result)
            print("已保存:", save_path)

    cap.release()
    cv2.destroyAllWindows()


# ==================== 程序入口 ====================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--image", type=str)
    group.add_argument("--video", type=str)
    group.add_argument("--camera", type=int)

    args = parser.parse_args()

    detector = CPUDetector()

    if args.image is not None:
        run_image(detector, args.image)

    elif args.video is not None:
        run_stream(detector, args.video)

    elif args.camera is not None:
        run_stream(detector, args.camera)


if __name__ == "__main__":
    main()
