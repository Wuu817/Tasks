import argparse
import json
import random
import time

import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--timeout", type=float)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

# 读取 YAML 或 JSON 配置
with open(args.config, encoding="utf-8") as f:
    if args.config.lower().endswith((".yaml", ".yml")):
        config = yaml.safe_load(f)
    elif args.config.lower().endswith(".json"):
        config = json.load(f)
    else:
        parser.error("配置文件必须是 .yaml、.yml 或 .json")

tasks = config["tasks"]
if args.timeout is not None:
    timeout = args.timeout
else:
    timeout = config.get("timeout", 15)

if args.seed is not None:
    random.seed(args.seed)

# status 记录已经处理过的任务及其结果
# pending 记录还没有处理的任务
status = {}
pending = tasks.copy()
start = time.monotonic()

while pending:
    # 在开始处理下一批任务前检查总时间
    if time.monotonic() - start >= timeout:
        for task in pending:
            status[task["name"]] = "TIMEOUT"
        break

    progressed = False

    for task in pending[:]:
        name = task["name"]
        dependencies = task["dependencies"]

        # 前置任务还没处理完，就先看其他任务
        if not all(dep in status for dep in dependencies):
            continue

        pending.remove(task)
        progressed = True

        # 只要有前置任务没成功，当前任务就跳过
        if any(status[dep] != "SUCCESS" for dep in dependencies):
            status[name] = "SKIPPED"
            print(name, "SKIPPED")
            continue

        # 最多尝试 3 次：第一次执行 + 最多两次重试
        for attempt in range(1, 4):
            time_left = timeout - (time.monotonic() - start)

            if time_left <= 0:
                status[name] = "TIMEOUT"
                break

            duration = task["duration"]

            # 本次任务所需时间已超过剩余时间
            if duration >= time_left:
                time.sleep(time_left)
                status[name] = "TIMEOUT"
                break

            time.sleep(duration)

            # 睡眠结束后再检查一次，以免已经超时
            if time.monotonic() - start >= timeout:
                status[name] = "TIMEOUT"
                break

            if random.random() < task["success_rate"]:
                status[name] = "SUCCESS"
                break

            if attempt == 3:
                status[name] = "FAILED"
            else:
                print(name, "RETRY")

        print(name, status[name])

        # 一旦超时，其余尚未处理的任务也不再执行
        if status[name] == "TIMEOUT":
            for remaining_task in pending:
                status[remaining_task["name"]] = "TIMEOUT"
            pending.clear()
            break

    # 一整轮都找不到可处理的任务，说明依赖关系有问题
    if not progressed:
        raise ValueError("任务依赖无法排好顺序：可能有依赖环或不存在的任务")

print("最终状态：", status)