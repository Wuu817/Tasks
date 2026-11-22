# M2-3 六类物品目标检测

## 项目简介
本项目基于YOLO11n，实现可口可乐、百事可乐、雪碧、杯子、纸巾和药膏六类物体的目标检测。
完成了自主数据采集与标注、模型训练、测试集评估、ONNX导出和CPU实时推理。

## 实验结果
- 数据集：940张图片，1551个标注目标
- 训练集：658张
- 验证集：188张
- 测试集：94张
- mAP@0.5：0.987
- mAP@0.5:0.95：0.891
- CPU平均帧率：29.96 FPS
- ONNX与PyTorch原始框坐标最大差异：0.00558472像素
- ONNX与PyTorch类别分数最大差异：0.00000989

## 项目主要文件

| 文件 | 功能 |
| train.py | 模型训练 |
| eval.py | 模型评估 |
| export_onnx.py | ONNX导出 |
| deploy_cpu.py | CPU推理 |
| benchmark.py | 性能测试 |
| check_consistency.py | 模型一致性验证 |
| models/best.pt | PyTorch模型 |
| models/best.onnx | ONNX模型 |
| docs/data.md | 数据报告 |
| docs/train.md | 训练报告 |
| docs/eval.md | 评估报告 |
| docs/deploy.md | 部署报告 |

数据集主要保存在`dataset/final_v2/`

## 环境

- Ubuntu 24.04
- Python 3.12
- PyTorch
- Ultralytics
- OpenCV
- ONNX Runtime

## 运行方法

进入项目目录并激活虚拟环境：
```bash
source .venv/bin/activate
```
训练：
```bash
python train.py
```
模型评估：
```bash
python eval.py
```
导出ONNX：
```bash
python export_onnx.py
```
一致性测试：
```bash
python check_consistency.py
```
CPU性能测试：
```bash
python benchmark.py
```
摄像头实时检测：
```bash
python deploy_cpu.py --camera 2
```
摄像头编号根据实际设备调整。各脚本可独立运行，运行前需确认项目数据和权重文件路径正确。
## 模型来源

Ultralytics YOLO11n

https://docs.ultralytics.com/models/yolo11/

使用官方预训练权重进行迁移学习，模型许可遵循Ultralytics对应条款。
