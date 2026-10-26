#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python3 sensor_analyzer.py
"""
sensor_analyzer.py  —— 上一届学长留下的"能用"的脚本

注释（学长原话）：
    "处理一下传感器数据就能用"

原本意图：
    1. 读取 sensor_data.csv（列：time, value）
    2. 计算 value 的平均值、标准差
    3. 剔除离群值（|value - mean| > 2 * std）
    4. 把清洗后的数据保存为 cleaned_data.csv
    5. 打印一份统计摘要

现状：跑不通 / 跑出来数不对。就交给你了。
"""

import csv
import os
import sys

INPUT_FILE = "sensor_data.csv"
OUTPUT_FILE = "cleaned_data.csv"

data = []
times = []
cleaned = []

print("=== 传感器数据分析 ===")

# --- 读取数据 ---
try:
    reader = csv.DictReader(open(INPUT_FILE, "r"))
except FileNotFoundError:
    print("错误：找不到输入文件", INPUT_FILE, file=sys.stderr)
    sys.exit(1)

if reader.fieldnames is not None and (
    "time" not in reader.fieldnames or "value" not in reader.fieldnames
):
    print("错误：CSV 必须包含 time 和 value 两列", file=sys.stderr)
    sys.exit(1)

for row in reader:
    try:
        t = float(row["time"])
        v = float(row["value"])
    except (ValueError, TypeError):
        print("错误：CSV 的 time 和 value 必须是数字", file=sys.stderr)
        sys.exit(1)

    times.append(t)
    data.append(v)

if not data:
    print("错误：输入文件中没有数据", file=sys.stderr)
    sys.exit(1)

print("共读取 %d 条数据" % len(data))

# --- 计算平均值 ---
total = 0
for v in data:
    total += v
mean = total / len(data)

# --- 计算标准差 ---
acc = 0
for v in data:
    acc += (v - mean) ** 2
std = (acc / len(data)) ** 0.5

# --- 剔除离群值 ---
for i in range(len(data)):
    v = data[i]
    if abs(v - mean) <= 2 * std:
        cleaned.append([times[i], v])

# --- 输出清洗后的数据 ---
output_path = OUTPUT_FILE
f = open(output_path, "w")
writer = csv.writer(f)
writer.writerow(["time", "value"])
for v in cleaned:
    writer.writerow(v)

print("均值 mean = %.4f" % mean)
print("标准差 std = %.4f" % std)
print("清洗后剩余 %d 条" % len(cleaned))
print("已保存到 %s" % output_path)
