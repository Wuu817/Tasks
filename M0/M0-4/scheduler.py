import argparse
import json
import random
import time

import yaml
from colorama import Fore, init

init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--timeout", type=float)
parser.add_argument("--seed", type=int)
parser.add_argument("--report", default="report.json")
args = parser.parse_args()

# 读取 YAML 或 JSON 配置
try:
    with open(args.config, encoding="utf-8") as f:
        if args.config.lower().endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif args.config.lower().endswith(".json"):
            config = json.load(f)
        else:
            parser.error("配置文件必须是 .yaml、.yml 或 .json")
except OSError as error:
    parser.error(f"无法读取配置文件 {args.config}: {error}")
except yaml.YAMLError as error:
    parser.error(f"YAML 配置文件格式错误 {args.config}: {error}")
except json.JSONDecodeError as error:
    parser.error(f"JSON 配置文件格式错误 {args.config}: {error}")

tasks = config["tasks"]
if not tasks:
    parser.error("任务列表不能为空：tasks 至少需要一个任务")

for task in tasks:
    rate = task["success_rate"]
    if rate < 0 or rate > 1:
        parser.error(f"任务 {task['name']} 的 success_rate 必须在 0 到 1 之间")

task_names = []
for task in tasks:
    name = task["name"]
    if name in task_names:
        parser.error(f"任务名重复：{name}")
    task_names.append(name)

for task in tasks:
    for dependency in task["dependencies"]:
        if dependency not in task_names:
            parser.error(
                f"任务 {task['name']} 依赖的任务 {dependency} 不存在"
            )

if args.timeout is not None:
    timeout = args.timeout
else:
    timeout = config.get("timeout", 15)

if args.seed is not None:
    random.seed(args.seed)

# status 记录已经处理过的任务及其结果
# pending 记录还没有处理的任务
status = {}
task_records = {}

for task in tasks:
    name = task["name"]
    task_records[name] = {
        "name": name,
        "status": None,
        "attempts": 0,
        "duration": 0.0,
        "started_at": None,
        "ended_at": None,
    }

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
            print(name, Fore.BLUE + "SKIPPED")
            continue

        # 最多尝试 3 次：第一次执行 + 最多两次重试
        for attempt in range(1, 4):
            time_left = timeout - (time.monotonic() - start)

            if time_left <= 0:
                status[name] = "TIMEOUT"
                break

            task_records[name]["attempts"] = attempt

            if task_records[name]["started_at"] is None:
                task_records[name]["started_at"] = time.time()
                task_run_start = time.monotonic()
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
                print(name, Fore.YELLOW + "RETRY")

        if task_records[name]["attempts"] > 0:
            task_records[name]["ended_at"] = time.time()
            task_records[name]["duration"] = time.monotonic() - task_run_start

        state = status[name]
        if state == "SUCCESS":
            print(name, Fore.GREEN + state)
        elif state == "FAILED":
            print(name, Fore.RED + state)
        else:
            print(name, state)

        # 一旦超时，其余尚未处理的任务也不再执行
        if status[name] == "TIMEOUT":
            for remaining_task in pending:
                status[remaining_task["name"]] = "TIMEOUT"
            pending.clear()
            break

    # 一整轮都找不到可处理的任务，说明依赖关系有问题
    if not progressed:
        parser.error("任务依赖形成环：剩余任务互相等待，无法执行")

for task in tasks:
    name = task["name"]
    task_records[name]["status"] = status[name]

report = {
    "timeout": "TIMEOUT" in status.values(),
    "total_duration": time.monotonic() - start,
    "tasks": list(task_records.values()),
}

with open(args.report, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("最终状态：", status)
