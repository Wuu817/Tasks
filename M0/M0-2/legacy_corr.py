import yaml
import csv
import math
import sys
import argparse

def calculate(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("两列数据长度必须相同")

    if len(xs) < 2:
        raise ValueError("至少需要两对数据")
    n = len(xs)


    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x = sum_x + xs[i]
        sum_y = sum_y + ys[i]

    mean_x = sum_x / n
    mean_y = sum_y / n

    dx = 0.0
    dy = 0.0
    prod = 0.0
    for i in range(n):
        a = xs[i] - mean_x
        b = ys[i] - mean_y
        dx = dx + a * a
        dy = dy + b * b
        prod = prod + a * b

    if dx == 0 or dy == 0:
        raise ValueError("某一列数据全部相同，无法计算相关系数")
    denom = math.sqrt(dx * dy)
    r = prod / denom
    return n, mean_x, mean_y, r

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config.yaml")
        args = parser.parse_args()
        CONFIG_PATH = args.config

        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        if not cfg:
            raise ValueError("配置文件为空")

        csv_path = cfg["input_csv"]
        col_x = cfg["columns"]["x"]
        col_y = cfg["columns"]["y"]

        xs = []
        ys = []

        with open(csv_path) as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV 文件为空")

            if col_x not in reader.fieldnames or col_y not in reader.fieldnames:
                raise ValueError("CSV 列名与配置不匹配")

            for row in reader:
                xs.append(float(row[col_x]))
                ys.append(float(row[col_y]))

        n, mean_x, mean_y, r = calculate(xs, ys)

        print("n =", n)
        print("mean_x =", mean_x)
        print("mean_y =", mean_y)
        print("r =", r)

    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print("错误：", error, file=sys.stderr)
        sys.exit(1)