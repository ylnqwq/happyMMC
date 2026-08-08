# IABC 人工蜂群算法实验项目

本项目用于对比人工蜂群算法及相关智能优化算法在单目标优化、多目标优化和微电网经济环境调度问题上的表现。代码包含算法实现、基准函数、批量实验、统计检验、结果导出和论文图表绘制脚本。

主要入口：

- 单目标算法对比：`single_objective/run_single_objective_comparison.py`
- 单目标 IABC 参数敏感性分析：`single_objective/run_iabc_sensitivity.py`
- 多目标基础算法对比：`multi_objective/run_multi_objective_standard_comparison.py`
- 多目标改进算法对比：`multi_objective/run_multi_objective_improved_comparison.py`
- 多目标 MOIABC 参数敏感性分析：`multi_objective/run_moiabc_elite_elimination_sensitivity.py`
- 多目标 MOIABC archive_rate 敏感性分析：`multi_objective/run_moiabc_archive_rate_sensitivity.py`
- 多目标 MOIABC 消融实验：`multi_objective/run_moiabc_ablation.py`
- 微电网调度应用案例：`multi_objective/application_point/run_microgrid_dispatch.py`

## 环境依赖

建议使用 Python 3.9 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

项目中的 Wilcoxon 配对符号秩检验为自行实现，不依赖 `scipy`。

## 项目结构

```text
IABC/
|-- experiment_utils.py
|-- README.md
|-- single_objective/
|   |-- run_single_objective_comparison.py
|   |-- run_iabc_sensitivity.py
|   |-- plot_single_objective_results.py
|   |-- single_objective_benchmarks.py
|   |-- statistical_tests.py
|   |-- so_utils.py
|   `-- algorithms/
|       |-- ABC.py
|       |-- ACO.py
|       |-- GA.py
|       |-- IABC.py
|       |-- IABC_MSS.py
|       `-- NDBP_ABC.py
`-- multi_objective/
    |-- run_multi_objective_comparison.py
    |-- run_multi_objective_standard_comparison.py
    |-- run_multi_objective_improved_comparison.py
    |-- run_moiabc_elite_elimination_sensitivity.py
    |-- run_moiabc_archive_rate_sensitivity.py
    |-- run_moiabc_ablation.py
    |-- plot_moiabc_sensitivity_results.py
    |-- multiobjective_benchmarks.py
    |-- statistical_tests.py
    |-- mo_utils.py
    |-- algorithms/
    |   |-- MOABC.py
    |   |-- ISSA.py
    |   |-- MODE.py
    |   |-- MOEAD.py
    |   |-- MOIABC.py
    |   |-- MOPSO.py
    |   |-- Yang_IGWO.py
    |   |-- Zhou_IMOABC.py
    |   `-- __init__.py
    `-- application_point/
        |-- microgrid_dispatch_model.py
        |-- run_microgrid_dispatch.py
        `-- results/
```

## 单目标优化实验

### 算法

单目标实验位于 `single_objective/`，当前包含 6 个算法：

- `ABC`：基本人工蜂群算法
- `GA`：遗传算法
- `ACO`：蚁群优化算法
- `IABC-MSS`：多策略综合改进人工蜂群算法
- `NDBP-ABC`：非确定性搜索与双向规划改进人工蜂群算法
- `IABC`：本项目单目标改进人工蜂群算法

### 测试函数

当前单目标测试集为 `CEC2022`，包含 `CEC2022_F1` 至 `CEC2022_F12`。默认维度为 10 维，边界为 `[-100, 100]`，并使用项目内置的 10 维平移、旋转和打乱数据。

### 默认配置

在 `single_objective/run_single_objective_comparison.py` 中配置：

```python
RUN_TIMES = 1
BOUNDS = [(-100, 100)] * 10
PARALLEL_WORKERS = 8
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = []
ENABLED_ALGORITHMS = []

COMMON_PARAMS = {
    "bee": 80,
    "max_iter": 800,
    "limit": 160,
}
```

`ENABLED_ALGORITHMS = []` 表示运行所有单目标算法。正式实验建议将 `RUN_TIMES` 调整为 30 或更高。

### 运行

```bash
python single_objective\run_single_objective_comparison.py
```

关闭单目标图像导出：

```powershell
$env:SO_SAVE_PLOTS="0"
python single_objective\run_single_objective_comparison.py
```

输出目录：

```text
single_objective/comparison_results/
```

主要输出：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：独立运行最优值曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线
- `wilcoxon_*_vs_iabc_results.csv`：各算法相对 `IABC` 的 Wilcoxon 检验
- `wilcoxon_test_results.csv`：Wilcoxon 汇总结果
- `average_rank_results.csv`：跨测试函数平均排名

### IABC 参数敏感性

```bash
python single_objective\run_iabc_sensitivity.py
```

默认扫描：

```python
RUN_TIMES = 30
SEED_BASE = 20240621
ELITE_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
ELIMINATION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
```

输出目录：

```text
single_objective/sensitivity_results/
```

主要输出：

- `sensitivity_detail_results.csv`
- `sensitivity_summary_by_function.csv`
- `sensitivity_average_rank.csv`

可使用 `plot_single_objective_results.py` 绘制对比表或敏感性分析图：

```bash
python single_objective\plot_single_objective_results.py --input-dir single_objective\comparison_results
python single_objective\plot_single_objective_results.py --mode sensitivity --input-dir single_objective\sensitivity_results
```

## 多目标优化实验

### 算法

多目标实验位于 `multi_objective/`，当前包含 7 个算法：

- `MOABC`：基本多目标人工蜂群算法
- `MO-DE`：多目标差分进化算法
- `MOEA/D`：基于分解的多目标进化算法
- `MOPSO`：多目标粒子群算法
- `Zhou-IMOABC`：Zhou 风格改进多目标人工蜂群算法
- `Yang-IGWO`：杨慧娴改进灰狼优化算法
- `ISSA`：改进麻雀搜索算法
- `MOIABC`：本项目多目标改进人工蜂群算法

### 测试函数

当前多目标测试集包括：

- `ZDT`：`ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
- `CEC2009_UF`：`UF1` 至 `UF10`
- `CEC2020_MMO`：`MMF1`, `MMF2`, `MMF4`, `MMF5`, `MMF7`, `MMF8`, `MMF10`, `MMF11`, `MMF12`, `MMF13`

多目标结果是一组 Pareto 非支配解，不是单个最优解。主要评价指标：

- `hypervolume`：超体积，越大越好
- `spacing`：解集分布间距，越小越好
- `best_sum`：档案中最小目标和，越小越好

### 默认配置

在 `multi_objective/run_multi_objective_comparison.py` 中配置：

```python
RUN_TIMES = env_int("MO_COMPARISON_RUN_TIMES", 30)
SEED_BASE = env_int("MO_COMPARISON_SEED_BASE", 20260723)
PARALLEL_WORKERS = env_int("MO_COMPARISON_WORKERS", 8)
SAVE_ARCHIVE_POINTS = env_bool("MO_COMPARISON_SAVE_ARCHIVE_POINTS", True)
SAVE_PLOTS = env_bool("MO_COMPARISON_SAVE_PLOTS", False)
SAVE_SUMMARY_PLOTS = env_bool("MO_COMPARISON_SAVE_SUMMARY_PLOTS", True)

ENABLED_SUITES = env_csv("MO_COMPARISON_SUITES", ["ZDT", "CEC2009_UF", "CEC2020_MMO"])
ENABLED_FUNCTION_IDS = env_csv("MO_COMPARISON_FUNCTION_IDS")

EXPERIMENT_GROUPS = [
    {
        "name": "standard_algorithms",
        "output_dir": MODULE_DIR / "mo_comparison_results_standard_algorithms",
        "algorithms": ["MO-DE", "MOEA/D", "MOPSO", "MOABC", "MOIABC"],
    },
    {
        "name": "improved_algorithms",
        "output_dir": MODULE_DIR / "mo_comparison_results_improved_algorithms",
        "algorithms": ["Zhou-IMOABC", "Yang-IGWO", "ISSA", "MOIABC"],
    },
]

COMMON_PARAMS = {
    "bee": 80,
    "max_iter": 800,
    "limit": 160,
    "archive_size": 100,
}

MOIABC_BEST_PARAMS = {
    "tournament_size": 3,
    "elite_rate": 0.25,
    "elimination_rate": 0.25,
    "archive_guidance_rate": 0.40,
}
```

### 运行

```bash
python multi_objective\run_multi_objective_standard_comparison.py
python multi_objective\run_multi_objective_improved_comparison.py
```

可用环境变量分批运行。示例：先跑 `ZDT` 和 `CEC2009_UF`，再跑 `MMF`（代码中对应测试集名为 `CEC2020_MMO`）：

```powershell
$env:MO_COMPARISON_SUITES="ZDT,CEC2009_UF"
$env:MO_COMPARISON_STANDARD_OUTPUT_DIR="mo_comparison_results_standard_zdt_uf"
python multi_objective\run_multi_objective_standard_comparison.py

$env:MO_COMPARISON_SUITES="CEC2020_MMO"
$env:MO_COMPARISON_STANDARD_OUTPUT_DIR="mo_comparison_results_standard_mmf"
python multi_objective\run_multi_objective_standard_comparison.py
```

改进算法组同理，把输出目录变量换成 `MO_COMPARISON_IMPROVED_OUTPUT_DIR`。

输出目录：

```text
multi_objective/mo_comparison_results_standard_algorithms/
multi_objective/mo_comparison_results_improved_algorithms/
```

主要输出：

- `*_results.csv`：每次独立运行的档案规模、目标和、spacing、hypervolume、耗时等
- `*_archive_points.csv`：Pareto 档案中的目标函数值
- `*_pareto_scatter.png`：Pareto 非支配解散点图（仅在 `MO_COMPARISON_SAVE_PLOTS=1` 时生成）
- `*_average_history.png`：平均收敛参考曲线（仅在 `MO_COMPARISON_SAVE_PLOTS=1` 时生成）
- `overall_average_history.png`：跨全部测试函数归一化平均后的总体收敛曲线
- `overall_average_history.csv`：总体收敛曲线数据
- `wilcoxon_*_vs_moiabc_results.csv`：启用 `MOIABC` 时生成的 Wilcoxon 检验
- `wilcoxon_test_results.csv`
- `average_rank_results.csv`

## 多目标 MOIABC 扩展实验

### elite_rate / elimination_rate 敏感性

```bash
python multi_objective\run_moiabc_elite_elimination_sensitivity.py
```

默认输出目录：

```text
multi_objective/moiabc_elite_elimination_sensitivity_results/
```

常用环境变量：

- `MOIABC_SENSITIVITY_RUN_TIMES`
- `MOIABC_SENSITIVITY_SEED_BASE`
- `MOIABC_SENSITIVITY_WORKERS`
- `MOIABC_SENSITIVITY_SUITES`
- `MOIABC_SENSITIVITY_FUNCTION_IDS`
- `MOIABC_ELITE_ELIMINATION_SENSITIVITY_OUTPUT_DIR`

示例：

```powershell
$env:MOIABC_SENSITIVITY_RUN_TIMES="5"
$env:MOIABC_SENSITIVITY_FUNCTION_IDS="ZDT1,UF1,MMF1"
python multi_objective\run_moiabc_elite_elimination_sensitivity.py
```

### archive_rate 敏感性

```bash
python multi_objective\run_moiabc_archive_rate_sensitivity.py
```

默认扫描：

```python
ARCHIVE_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
```

默认输出目录：

```text
multi_objective/moiabc_archive_rate_sensitivity_results/
```

常用环境变量：

- `MOIABC_ARCHIVE_RATE_SENSITIVITY_RUN_TIMES`
- `MOIABC_ARCHIVE_RATE_SENSITIVITY_SEED_BASE`
- `MOIABC_ARCHIVE_RATE_SENSITIVITY_WORKERS`
- `MOIABC_ARCHIVE_RATE_SENSITIVITY_SUITES`
- `MOIABC_ARCHIVE_RATE_SENSITIVITY_FUNCTION_IDS`
- `MOIABC_ARCHIVE_RATE_SENSITIVITY_OUTPUT_DIR`

### 消融实验

```bash
python multi_objective\run_moiabc_ablation.py
```

默认消融变体：

- `MOIABC`
- `MOIABC-no-good-point-init`
- `MOIABC-no-tournament-selection`
- `MOIABC-no-elite-enhancement`
- `MOIABC-no-worst-elimination`
- `MOABC-equivalent`

默认输出目录：

```text
multi_objective/moiabc_ablation_results/
```

常用环境变量：

- `MOIABC_ABLATION_RUN_TIMES`
- `MOIABC_ABLATION_SEED_BASE`
- `MOIABC_ABLATION_WORKERS`
- `MOIABC_ABLATION_SUITES`
- `MOIABC_ABLATION_FUNCTION_IDS`
- `MOIABC_ABLATION_VARIANTS`
- `MOIABC_ABLATION_SAVE_ARCHIVE_POINTS`
- `MOIABC_ABLATION_OUTPUT_DIR`

### 绘图

`plot_moiabc_sensitivity_results.py` 可用于绘制 MOIABC 参数敏感性和消融实验图表：

```bash
python multi_objective\plot_moiabc_sensitivity_results.py --input-dir multi_objective\moiabc_elite_elimination_sensitivity_results
python multi_objective\plot_moiabc_sensitivity_results.py --mode ablation --input-dir multi_objective\moiabc_ablation_results
```

## 微电网调度应用案例

应用案例位于 `multi_objective/application_point/`，使用 MOIABC 对 24 小时微电网调度进行双目标优化。

决策变量：

- 24 小时柴油机出力
- 24 小时储能充放电功率

目标函数：

- 经济成本
- 环境治理成本

运行命令：

```bash
python multi_objective\application_point\run_microgrid_dispatch.py
```

常用环境变量：

- `APP_SEED`
- `APP_BEE`
- `APP_MAX_ITER`
- `APP_LIMIT`
- `APP_ARCHIVE_SIZE`
- `APP_SAVE_PLOTS`

输出目录：

```text
multi_objective/application_point/results/
```

主要输出：

- `multi_objective_summary.json`
- `multi_objective_pareto.csv`
- `multi_objective_compromise_dispatch.csv`
- `multi_objective_pareto.png`
- `multi_objective_history.png`

## 常用配置方法

只运行指定测试函数：

```python
ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
```

只运行指定单目标算法：

```python
ENABLED_ALGORITHMS = ["ABC", "IABC"]
```

只运行指定多目标测试函数：

```python
ENABLED_FUNCTION_IDS = ["ZDT1", "UF1", "MMF1"]
```

只运行指定多目标算法：

```python
ENABLED_ALGORITHMS = ["MOABC", "MOIABC"]
```

关闭输出图像或档案点导出时，按脚本实际支持方式修改：

- 单目标对比：使用环境变量 `SO_SAVE_PLOTS=0`
- 微电网案例：使用环境变量 `APP_SAVE_PLOTS=0`
- 多目标主对比：使用环境变量 `MO_COMPARISON_SAVE_PLOTS=0`、`MO_COMPARISON_SAVE_SUMMARY_PLOTS=0`、`MO_COMPARISON_SAVE_ARCHIVE_POINTS=0`
- 多目标消融：使用环境变量 `MOIABC_ABLATION_SAVE_ARCHIVE_POINTS=0`

## 统计结果说明

`wilcoxon_test_results.csv` 常用字段：

- `wins`, `ties`, `losses`：改进算法相对基准算法的胜、平、负次数
- `mean_difference`：按指标方向换算后的平均差值，正值表示改进算法更好
- `p_two_sided`：双侧检验 p 值
- `p_improved`：改进方向单侧检验 p 值
- `significant_0_05`：双侧检验是否达到 0.05 显著性水平

`average_rank_results.csv` 和各类 `*_average_rank.csv` 中的 `average_rank` 越小，表示整体排名越靠前。

## 注意事项

- 单目标 CEC2022 当前使用项目内置的 10 维平移、旋转和打乱数据；如果修改维度，需要补充对应维度的数据。
- `RUN_TIMES`、`bee`、`max_iter`、`limit`、测试函数数量和算法数量会直接影响运行时间。
- CSV 使用 `utf-8-sig` 编码保存，便于用 Excel 打开。
- 图像中文字体依赖系统字体，脚本默认尝试使用 `Microsoft YaHei`、`SimHei`、`SimSun`。
- `.gitignore` 已忽略实验输出目录、Python 缓存、本地环境文件和常见密钥文件。
