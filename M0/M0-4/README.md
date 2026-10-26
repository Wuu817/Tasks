# M0-4 TaskScheduler 实现记录（初稿）

## 完成的任务
编写了一个终端任务调度模拟器。程序从 YAML 或 JSON 配置读取任务，按依赖关系安排顺序，用设定的成功概率模拟每次尝试，处理失败重试、下游跳过和全局超时，最后输出 JSON 报告。
项目中使用 `scheduler.py` 作为入口，`tasks.yaml` 作为 7 个任务的示例配置，`requirements.txt` 记录第三方依赖，`report.json` 保存一次运行的报告样例。

## 实现过程

### 环境与配置

我用 `uv` 在本题文件夹中创建 `.venv`，安装 `PyYAML` 和 `colorama`。`json`、`argparse`、`random`、`time` 来自 Python 标准库。程序接收 `--config` 后，按文件扩展名选择 `yaml.safe_load()` 或 `json.load()`；使用 `--timeout` 时，它覆盖配置文件中的全局 `timeout`。`--seed` 让随机判定顺序可以复现。
### 依赖调度
`pending` 保存还未处理的任务，`status` 记录已确定的状态，`task_records` 保存报告需要的详细记录。每轮扫描 `pending`：前置任务尚未出结果时先等待；前置任务全部成功才执行当前任务；有前置任务失败或被跳过，就把当前任务标记为 `SKIPPED`。因此不依赖失败任务的分支仍会继续。
每轮的 `progressed` 只表示“这一轮是否处理了至少一个任务”。即使独立任务先完成，只要还有互相等待的任务，下一轮仍会重新检查；整轮没有进展时报告依赖环。调度前还检查了依赖名是否存在、任务名是否重复。
### 重试、计时与输出
每次尝试用 `time.sleep(duration)` 模拟耗时，用 `random.random() < success_rate` 判断是否成功。一次失败后最多重试两次，总尝试次数不超过 3；最终失败记为 `FAILED`。我用 `time.monotonic()` 计算全局经过时间，在任务执行前后检查上限；超时后剩余任务记为 `TIMEOUT`。
终端实时显示状态：`SUCCESS` 绿色、`FAILED` 红色、`RETRY` 黄色、`SKIPPED` 蓝色。报告包含 `timeout`、`total_duration`，以及每个任务的 `name`、`status`、`attempts`、`duration`、`started_at`、`ended_at`。未开始任务的时间字段为 `null`。
### 输入错误处理
测试过程中，我发现文件不存在、YAML/JSON 格式错误最初会显示 `Traceback`，于是为读取阶段加入对应错误提示。空任务列表原来会被当成成功运行；`success_rate` 超出 0～1 原来可能被误判；缺失依赖，依赖环，重名任务也需要在运行前或调度时明确报错。我逐项复现并修改了这些问题。
## 验证
- 用项目自带的 `tasks.yaml` 检查正常执行、重试、失败和跳过。
- 用一份新的 7 任务 YAML 检查分叉、失败重试、下游跳过与独立分支继续执行。
- 用一份新的 5 任务 JSON 检查配置顺序打乱后的依赖顺序和汇合。
- 用 `--timeout 0.1` 检查全局超时；报告中的 `timeout` 为 `true`，未执行任务仍保留在报告中。
- 用错误配置检查文件不存在，YAML/JSON语法错误，空列表，成功率越界，缺失依赖，依赖环和重名任务。

## 命令行参数

| 参数 | 是否必填 | 作用 |
| `--config 路径` | 是 | 指定 `.yaml`、`.yml` 或 `.json` 配置文件 |
| `--timeout 秒数` | 否 | 覆盖配置中的**全局**时间上限，不是单个任务的上限 |
| `--report 路径` | 否 | 指定 JSON 报告路径，默认是 `report.json` |
| `--seed 整数` | 否 | 固定随机数序列，方便重复观察同一组随机判定 |

## 三个运行示例

以下命令都在 `M0/M0-4` 文件夹中运行，使用本题自带的 `tasks.yaml`。三个示例使用同一份配置，通过不同参数展示正常调度、另一组随机结果和全局超时。

1. 固定随机种子为 42，使用配置中的 15 秒总时限，将结果写入默认的 `report.json`：

   ```bash
   .venv/bin/python scheduler.py --config tasks.yaml --seed 42
   ```

   本次验证中，`AvoidObstacle` 最终为 `FAILED`，依赖它的 `Place` 和 `ReturnHome` 为 `SKIPPED`。

2. 改用随机种子 1，并把结果写入另一个报告文件：

   ```bash
   .venv/bin/python scheduler.py --config tasks.yaml --seed 1 --report report_seed1.json
   ```

   本次验证中，七个任务最终全部为 `SUCCESS`。

3. 将整个调度过程的时限设为 0.1 秒，检查超时处理：

   ```bash
   .venv/bin/python scheduler.py --config tasks.yaml --timeout 0.1 --seed 42 --report report_timeout.json
   ```

   本次验证中，报告的 `timeout` 为 `true`，七个任务都记录为 `TIMEOUT`。