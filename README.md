# MOIABC 多目标优化实验

本项目用于评估改进人工蜂群算法 MOIABC 在多目标优化问题上的表现，包含基准函数对比、参数敏感度分析、消融实验、统计检验，以及微电网经济-环境调度应用。

## 环境

建议使用 Python 3.9 或更高版本。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

## 项目结构

```text
MOIABC/
|-- multi_objective/
|   |-- algorithms/                         # MOIABC 与对比算法
|   |-- application_point/                  # 微电网调度应用
|   |-- experiment_utils.py                 # 实验参数与 CSV 工具
|   |-- mo_utils.py                         # Pareto 与指标计算
|   |-- multiobjective_benchmarks.py        # ZDT、UF、MMF 测试函数
|   |-- run_multi_objective_comparison.py   # 通用对比实验逻辑
|   |-- run_multi_objective_standard_comparison.py
|   |-- run_multi_objective_improved_comparison.py
|   |-- run_moiabc_elite_elimination_sensitivity.py
|   |-- run_moiabc_archive_rate_sensitivity.py
|   |-- run_moiabc_ablation.py
|   |-- rebuild_multi_objective_statistics.py
|   `-- plot_*.py                           # 结果绘图脚本
|-- 结果/                                    # 已导出的实验结果
`-- 旧结果/                                  # 历史结果备份
```

## 基准函数与算法

支持 25 个双目标基准函数：

- `ZDT`：`ZDT1`、`ZDT2`、`ZDT3`、`ZDT4`、`ZDT6`
- `CEC2009_UF`：`UF1` 至 `UF10`
- `CEC2020_MMO`：`MMF1`、`MMF2`、`MMF4`、`MMF5`、`MMF7`、`MMF8`、`MMF10` 至 `MMF13`

对比算法包括 `MOABC`、`MO-DE`、`MOEA/D`、`MOPSO`、`Zhou-IMOABC`、`Yang-IGWO`、`ISSA` 和 `MOIABC`。结果以 Pareto 非支配解集表示，主要评价指标为超体积（`hypervolume`，越大越好）、间距（`spacing`，越小越好）和 IGD/IGD+（越小越好）。

## 实验设计

每一次独立运行使用固定种子生成初始种群，算法结束后保留最终 Pareto 档案。默认配置如下：

| 项目 | 默认值 |
|---|---:|
| 独立运行次数 | 30 |
| 随机种子基数 | 20260723 |
| 并行进程数 | 8 |
| 种群/蜂群规模 | 80 |
| 最大迭代次数 | 800 |
| `limit` | 160 |
| Pareto 档案容量 | 100 |
| 锦标赛规模 | 3 |
| MOIABC 精英率 | 0.25 |
| MOIABC 淘汰率 | 0.25 |
| MOIABC 档案引导率 | 0.40 |

标准算法组比较 `MO-DE`、`MOEA/D`、`MOPSO`、`MOABC` 与 `MOIABC`；改进算法组比较 `Zhou-IMOABC`、`Yang-IGWO`、`ISSA` 与 `MOIABC`。两组均在同一批 25 个函数上重复运行，便于分别回答“相对经典方法”和“相对改进方法”的表现问题。

### 指标说明

- `hypervolume`：非支配解集支配的目标空间体积，使用每个测试函数定义的参考点计算，数值越大代表收敛性和覆盖性越好。
- `spacing`：每个非支配解到其最近邻的距离标准差，越小表示解集分布越均匀。
- `best_sum`：档案中 `f1 + f2` 的最小值，仅用于提供一个可读的折中标量，不能替代 Pareto 前沿本身。
- `IGD` 与 `IGD+`：先汇总同一函数、同一批次所有算法与所有重复运行的档案点，提取经验非支配前沿，再对目标值归一化后计算。因此两项指标用于同一批实验内部横向比较，不应与不同批次的绝对数值直接比较。
- `average_rank`：每个函数上按指标优劣排序后的平均名次，数值越小越好；`best_count` 是该算法取得该指标第一名的函数数。

Wilcoxon 结果使用配对的独立运行数据，`p_two_sided < 0.05` 表示双侧检验显著。Friedman 排名文件用于比较多个算法或多个参数配置的整体排序。

## 运行实验

在项目根目录执行以下命令。

标准算法对比：

```powershell
python multi_objective\run_multi_objective_standard_comparison.py
```

改进算法对比：

```powershell
python multi_objective\run_multi_objective_improved_comparison.py
```

MOIABC `elite_rate` / `elimination_rate` 敏感度分析：

```powershell
python multi_objective\run_moiabc_elite_elimination_sensitivity.py
```

MOIABC `archive_rate` 敏感度分析：

```powershell
python multi_objective\run_moiabc_archive_rate_sensitivity.py
```

MOIABC 消融实验：

```powershell
python multi_objective\run_moiabc_ablation.py
```

微电网调度应用：

```powershell
python multi_objective\application_point\run_microgrid_dispatch.py
```

> 注意：对比实验、敏感度分析和消融实验会重建其目标结果目录；同名目录中的旧文件会被清除。需要保留现有结果时，先通过相应的 `*_OUTPUT_DIR` 环境变量指定新目录。

## 常用配置

实验脚本通过环境变量控制运行规模、测试集和输出位置。例如，仅运行少量函数以快速验证：

```powershell
$env:MO_COMPARISON_RUN_TIMES = "3"
$env:MO_COMPARISON_SUITES = "ZDT"
$env:MO_COMPARISON_FUNCTION_IDS = "ZDT1,ZDT2"
$env:MO_COMPARISON_SAVE_PLOTS = "0"
python multi_objective\run_multi_objective_standard_comparison.py
```

常用变量如下：

| 用途 | 环境变量 |
|---|---|
| 对比实验重复次数 | `MO_COMPARISON_RUN_TIMES` |
| 并行进程数 | `MO_COMPARISON_WORKERS` |
| 测试集 | `MO_COMPARISON_SUITES` |
| 测试函数 | `MO_COMPARISON_FUNCTION_IDS` |
| 保存 Pareto 图 | `MO_COMPARISON_SAVE_PLOTS` |
| 保存档案点 | `MO_COMPARISON_SAVE_ARCHIVE_POINTS` |
| 标准组输出目录 | `MO_COMPARISON_STANDARD_OUTPUT_DIR` |
| 改进组输出目录 | `MO_COMPARISON_IMPROVED_OUTPUT_DIR` |

敏感度和消融实验的主要变量如下：

| 实验 | 重复次数 | 测试集/函数 | 输出目录 | 参数筛选 |
|---|---|---|---|---|
| 精英率-淘汰率敏感度 | `MOIABC_SENSITIVITY_RUN_TIMES` | `MOIABC_SENSITIVITY_SUITES`、`MOIABC_SENSITIVITY_FUNCTION_IDS` | `MOIABC_ELITE_ELIMINATION_SENSITIVITY_OUTPUT_DIR` | 无 |
| 档案引导率敏感度 | `MOIABC_ARCHIVE_RATE_SENSITIVITY_RUN_TIMES` | `MOIABC_ARCHIVE_RATE_SENSITIVITY_SUITES`、`MOIABC_ARCHIVE_RATE_SENSITIVITY_FUNCTION_IDS` | `MOIABC_ARCHIVE_RATE_SENSITIVITY_OUTPUT_DIR` | `MOIABC_ARCHIVE_RATE_SENSITIVITY_ARCHIVE_RATES` |
| 消融实验 | `MOIABC_ABLATION_RUN_TIMES` | `MOIABC_ABLATION_SUITES`、`MOIABC_ABLATION_FUNCTION_IDS` | `MOIABC_ABLATION_OUTPUT_DIR` | `MOIABC_ABLATION_VARIANTS` |

精英率-淘汰率实验默认扫描 `0.05, 0.10, 0.15, 0.20, 0.25` 的 25 种组合；档案引导率实验默认扫描 `0.10, 0.20, 0.30, 0.40, 0.50`。微电网应用使用 `APP_RUN_TIMES`、`APP_WORKERS`、`APP_ALGORITHMS`、`APP_SAVE_PLOTS` 与 `APP_OUTPUT_DIR`。

完整的单函数验证示例：

```powershell
$env:MO_COMPARISON_RUN_TIMES = "3"
$env:MO_COMPARISON_WORKERS = "1"
$env:MO_COMPARISON_SUITES = "ZDT"
$env:MO_COMPARISON_FUNCTION_IDS = "ZDT1"
$env:MO_COMPARISON_SAVE_PLOTS = "1"
$env:MO_COMPARISON_STANDARD_OUTPUT_DIR = "quickcheck_zdt1"
python multi_objective\run_multi_objective_standard_comparison.py
```

## 输出结果

默认结果保存在 `multi_objective/` 下对应的结果目录中：

- `mo_comparison_results_standard_algorithms/`
- `mo_comparison_results_improved_algorithms/`
- `moiabc_elite_elimination_sensitivity_results/`
- `moiabc_archive_rate_sensitivity_results/`
- `moiabc_ablation_results/`
- `application_point/results/`

### 基准对比结果

当前版本将结果汇总为以下文件，而不是为每个函数分散保存多个中间文件：

| 文件 | 内容 |
|---|---|
| `mo_comparison_summary_by_function.csv` | 每个函数、每种算法的 HV、Spacing、Best Sum、IGD、IGD+ 的均值与标准差 |
| `igd_igd_plus_results.csv` | 每次独立运行的 IGD 与 IGD+ 明细 |
| `average_rank_results.csv` | 每项指标的平均排名和第一名次数 |
| `wilcoxon_test_results.csv` | MOIABC 对各基线算法的配对 Wilcoxon 检验 |
| `mo_comparison_*_table.csv/png` | HV、Spacing、Best Sum、IGD、IGD+ 的论文用汇总表与图 |
| `mo_comparison_friedman_*_rank.csv/png` | 多算法整体排名的 Friedman 统计结果 |

`summary_by_function` 的字段以 `mean_*` 和 `std_*` 开头，分别表示 30 次独立运行的均值和标准差。逐次结果还包含 `seed`、`archive_size`、`time` 和经验参考前沿点数，便于复现或排查异常运行。

### 参数与消融结果

敏感度结果目录包含 `*_detail_results.csv`（逐次数据）、`*_summary_by_function.csv`（函数级汇总）、`*_average_rank.csv`（参数组合排名）及 `*_friedman_*.csv/png`。消融实验额外输出：

- `moiabc_ablation_effect_summary.csv`：各模块去除后相对完整 MOIABC 的指标变化；
- `wilcoxon_*_vs_moiabc_results.csv`：每个变体与完整算法的配对检验；
- `moiabc_ablation_*_table.csv/png`：HV、Spacing、Best Sum、IGD、IGD+ 的可直接用于论文的表格和图。

### 微电网应用结果

`application_point/results/` 保存调度结果及跨算法对比：

| 文件 | 内容 |
|---|---|
| `multi_objective_summary.json` | MOIABC 配置、选中运行、指标、折中解和储能状态摘要 |
| `multi_objective_metrics.csv` / `multi_objective_metrics_summary.csv` | 30 次运行的指标明细及统计值 |
| `multi_objective_pareto.csv` / `multi_objective_reference_pareto.csv` | 当前档案与经验参考 Pareto 前沿 |
| `multi_objective_compromise_dispatch.csv` | 折中解的分时功率调度 |
| `microgrid_algorithm_comparison.csv` | MOIABC、MOABC、MOPSO、MOEA/D、MO-DE 的可比指标 |
| `algorithm_*` 图表和 CSV | 平均 Pareto 前沿、收敛曲线、IGD/IGD+ 与调度功率曲线 |

CSV 使用 `utf-8-sig` 编码，可直接用 Excel 打开。

## 当前保存结果摘要

下列内容来自仓库中当前的结果文件，配置为 25 个基准函数、30 次独立运行、种群规模 80、800 次迭代。这是当前实验快照，不替代重新运行后的统计结论。

| 实验 | 当前结果摘要 | 来源 |
|---|---|---|
| 标准算法对比 | MOIABC 的 HV 平均排名为 `1.44`，在 25 个函数中获得 15 次 HV 第一；IGD 平均排名为 `1.56`，获 13 次第一；IGD+ 平均排名为 `1.48`，获 14 次第一。 | `mo_comparison_results_standard_algorithms/average_rank_results.csv` |
| 改进算法对比 | MOIABC 的 HV 平均排名为 `1.16`，获 21 次第一；IGD+ 平均排名为 `1.36`，获 17 次第一。 | `mo_comparison_results_improved_algorithms/average_rank_results.csv` |
| 精英率-淘汰率敏感度 | `elite_rate=0.25`、`elimination_rate=0.15` 的综合平均排名最低，为 `9.64`；该扫描使用 25 个参数组合和 3 项排序指标。 | `moiabc_elite_elimination_sensitivity_results/moiabc_sensitivity_average_rank.csv` |
| 档案引导率敏感度 | `archive_rate=0.10` 的综合平均排名最低，为 `2.7067`；`0.40` 次之，为 `2.8667`。 | `moiabc_archive_rate_sensitivity_results/moiabc_archive_rate_sensitivity_average_rank.csv` |
| 微电网 MOIABC | 30 次运行的平均 HV 为 `1.20999993`，平均 IGD 为 `2.6849e-08`；选中第 29 次运行的折中解经济成本为 `159.4872`、环境成本为 `22.5793`、可再生能源利用率为 `100%`。 | `application_point/results/multi_objective_summary.json` |

消融结果需要按指标阅读：当前快照中，完整 MOIABC 在 Best Sum 上与“去除优良点初始化”并列最低平均排名 `3.20`；而“去除最差解淘汰”在 HV、IGD 和 IGD+ 上的平均排名分别为 `7.00`、`7.32` 和 `7.08`，表现最弱。完整结论应同时参考效应汇总和 Wilcoxon 检验，不宜仅凭单个指标下判断。

## 绘图与统计

```powershell
python multi_objective\plot_multi_objective_comparison_results.py --help
python multi_objective\plot_moiabc_sensitivity_results.py --help
python multi_objective\plot_multi_objective_igd_results.py --help
python multi_objective\rebuild_multi_objective_statistics.py --help
```

使用 `--help` 查看各绘图或统计脚本支持的输入目录、模式和筛选参数。
