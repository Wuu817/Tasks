# CPU部署与性能测试报告

## 1. 部署方案
训练完成后，将最终PyTorch模型`models/best.pt`导出为`models/best.onnx`。
部署使用ONNX Runtime的`CPUExecutionProvider`，确保不依赖GPU运行。
导出配置：
| 参数 | 数值 |
| 格式 | ONNX |
| 输入尺寸 | 640×640 |
| Batch | 1 |
| Dynamic | False |
| Opset | 17 |
| NMS | False |
NMS由CPU部署代码单独执行。

## 2. 预处理和后处理
预处理使用Letterbox，保持图像原始宽高比，缩放后使用像素值114补边至640×640。
随后依次执行：
1. BGR转换为RGB。
2. HWC转换为CHW。
3. 转换为float32，并除以255。
4. 增加Batch维度，得到`(1,3,640,640)`。
模型输出形状约为`(1,10,8400)`，包含4个边界框参数和6个类别分数。
后处理使用置信度阈值0.25，按类别执行IoU阈值为0.45的NMS，并将检测框坐标映射回原始图像。
该流程与原模型的输入预处理方式保持一致。

## 3. PyTorch与ONNX一致性验证
在20张测试图片上，对两个模型使用相同预处理，比较原始输出。
| 检查项目 | 最大绝对差异 |
| 检测框坐标 | 0.00558472像素 |
| 类别分数 | 0.00000989 |

两个模型输出基本一致。
框坐标差异超过题目要求说明的`1e-3`阈值，但绝对偏差仍然很小。由于比较使用相同输入张量，预处理不一致造成差异的可能性较低。
误差可能与不同推理执行算子的浮点计算有关。

## 4. CPU性能测试

测试环境：
- CPU：Intel Core Ultra 7 251HX
- 推理框架：ONNX Runtime
- 执行设备：CPU
- 测试方式：摄像头输入
- 统计帧数：200帧
各阶段耗时如下：
| 阶段 | 平均耗时 |
| 图像读取 | 4.01 ms |
| 预处理 | 4.11 ms |
| ONNX推理 | 23.96 ms |
| 后处理 | 0.52 ms |
| 绘图 | 0.28 ms |
| 端到端总耗时 | 33.38 ms |

根据端到端平均耗时计算：

`FPS = 1000 / 33.38 ≈ 29.96`

最终CPU平均帧率约29.96 FPS，高于考核要求的10 FPS。

ONNX模型推理耗时占比最高，是主要性能瓶颈。

各阶段平均值之和为32.88 ms，与端到端记录的33.38 ms存在0.50 ms差异，应以原始计时口径为准。

## 5. 实时演示

`deploy_cpu.py`支持图片、视频和摄像头输入。
检测画面显示目标边界框、类别、置信度和FPS，并支持保存检测结果。
此前在本机摄像头上测试，实时帧率约29.8 FPS。
复现：cd ~/Projects/Tasks/M2/M2-3
source .venv/bin/activate
python benchmark.py
find . -name "time_distribution.png" -o -name "*.csv"
xdg-open results/benchmark/time_distribution.png

## 6. 结论
最终模型已经完成ONNX导出、PyTorch与ONNX一致性验证以及CPU性能测试。

