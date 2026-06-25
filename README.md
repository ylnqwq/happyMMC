# IABC人工蜂群算法实验项目

本项目用于比较人工蜂群算法及相关智能优化算法在单目标优化和多目标优化测试函数上的表现。项目包含算法实现、测试函数、批量实验、统计检验、结果导出、图像绘制和 IABC 参数敏感性分析。

主要入口：

- 单目标实验：`single_objective/run_single_objective_comparison.py`
- 多目标实验：`multi_objective/run_multi_objective_comparison.py`
- IABC 参数敏感性分析：`single_objective/run_iabc_sensitivity.py`

## 当前实验内容

### 单目标优化

单目标部分位于 `single_objective/`，当前包含 6 个算法：

- `ABC`：基本人工蜂群算法
- `GA`：遗传算法
- `ACO`：蚁群优化算法
- `IABC-MSS`：多策略综合改进人工蜂群算法
- `NDBP-ABC`：非确定性搜索与双向规划改进人工蜂群算法
- `IABC`：本项目的单目标改进人工蜂群算法

当前默认配置：

- `RUN_TIMES = 1`
- 维度：10 维
- 边界：`[-100, 100]`
- 默认测试集：`CEC2022`
- 默认算法：全部单目标算法
- 公共参数：`bee=75, max_iter=750, limit=150`
- IABC 参数：`tournament_size=3, elite_rate=0.25, elimination_rate=0.15`

当前单目标测试函数：

- `CEC2022`：`CEC2022_F1` 到 `CEC2022_F12`

单目标统计指标：

- `best_value`：目标函数值，越小越好
- `error`：与理论最优值的误差，越小越好
- Wilcoxon 配对符号秩检验：默认以 `IABC` 为改进算法，与其他已启用算法比较
- 平均排名：跨测试函数统计各算法整体排名

### 多目标优化

多目标部分位于 `multi_objective/`，当前包含 6 个算法：

- `MOABC`：基本多目标人工蜂群算法
- `NSGA-II`：基本非支配排序遗传算法
- `MOPSO`：基本多目标粒子群算法
- `Zhou-IMOABC`：Zhou 风格改进多目标人工蜂群算法
- `Zhao-IMOABC`：Zhao 风格改进多目标人工蜂群算法
- `MOIABC`：本项目的多目标改进人工蜂群算法

当前默认配置：

- `RUN_TIMES = 10`
- 默认测试集：`ZDT` 和完整 `CEC2020_MMO`
- 默认算法：`MOABC`, `MOIABC`
- 公共参数：`bee=75, max_iter=750, limit=150, archive_size=100`
- MOIABC 参数：`tournament_size=3, elite_rate=0.05, elimination_rate=0.10`

当前多目标测试函数：

- `ZDT`：`ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
- `CEC2020_MMO`：完整启用 24 个 MMF 测试函数，包括 `MMF1`, `MMF2`, `MMF4`, `MMF5`, `MMF7`, `MMF8`, `MMF10` 到 `MMF16_L3` 等

多目标统计指标：

- `hypervolume`：超体积，越大越好
- `spacing`：间距指标，越小越好
- `best_sum`：最小目标和，越小越好
- Wilcoxon 配对符号秩检验：默认以 `MOIABC` 为改进算法，与其他已启用算法比较
- 平均排名：跨测试函数统计各算法整体排名

多目标结果是一组 Pareto 非支配解，不是单个最优解。

## 项目结构

```text
IABC/
|-- single_objective/
|   |-- run_single_objective_comparison.py              # 单目标实验入口
|   |-- run_iabc_sensitivity.py    # IABC 参数敏感性分析
|   |-- single_objective_benchmarks.py   # CEC2022 单目标测试函数
|   |-- embedded_cec_data.py             # 10 维 CEC 平移、旋转和打乱数据
|   |-- statistical_tests.py             # Wilcoxon 与平均排名统计
|   |-- algorithms/
|   |   |-- ABC.py
|   |   |-- GA.py
|   |   |-- ACO.py
|   |   |-- IABC.py
|   |   |-- IABC_MSS.py
|   |   `-- NDBP_ABC.py
|   `-- comparison_results/              # 单目标实验输出目录
|
|-- multi_objective/
|   |-- run_multi_objective_comparison.py          # 多目标实验入口
|   |-- multiobjective_benchmarks.py     # ZDT / CEC2020 MMO 多目标测试函数
|   |-- mo_utils.py                      # Pareto 排序、拥挤距离、档案维护等工具
|   |-- statistical_tests.py             # Wilcoxon 与平均排名统计
|   |-- algorithms/
|   |   |-- MOABC.py
|   |   |-- MOIABC.py
|   |   |-- MOPSO.py
|   |   |-- NSGA2.py
|   |   |-- Zhou_IMOABC.py
|   |   `-- Zhao_IMOABC.py
|   `-- mo_comparison_results/           # 多目标实验输出目录
|
|-- .gitignore
`-- README.md
```

## 环境依赖

建议使用 Python 3.9 或更高版本。

安装依赖：

```bash
pip install numpy matplotlib
```

使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

项目中的 Wilcoxon 检验为自行实现，不依赖 `scipy`。

## 运行方法

在项目根目录运行单目标实验：

```bash
python single_objective\run_single_objective_comparison.py
```

Windows PowerShell 4-core server command:

```powershell
$env:SO_SAVE_PLOTS="0"
python single_objective\run_single_objective_comparison.py
```

在项目根目录运行多目标实验：

```bash
python multi_objective\run_multi_objective_comparison.py
```

在多核服务器上运行多目标实验时，可以开启并行并关闭图像和档案点导出以减少运行时间和磁盘 I/O：

```bash
set MO_SAVE_PLOTS=0
set MO_SAVE_ARCHIVE_POINTS=0
python multi_objective\run_multi_objective_comparison.py
```

Linux 服务器可使用：

```bash
MO_SAVE_PLOTS=0 MO_SAVE_ARCHIVE_POINTS=0 python multi_objective/run_multi_objective_comparison.py
```

运行 IABC 参数敏感性分析：

```bash
python single_objective\run_iabc_sensitivity.py
```

也可以进入子目录后运行：

```bash
cd single_objective
python run_single_objective_comparison.py
```

```bash
cd multi_objective
python run_multi_objective_comparison.py
```

## 参数配置

### 单目标参数

在 `single_objective/run_single_objective_comparison.py` 中修改：

```python
RUN_TIMES = 1
BOUNDS = [(-100, 100)] * 10
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = []
ENABLED_ALGORITHMS = []

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
}
```

### 多目标参数

在 `multi_objective/run_multi_objective_comparison.py` 中修改：

```python
RUN_TIMES = 10
ENABLED_SUITES = ["ZDT", "CEC2020_MMO"]
ENABLED_FUNCTION_IDS = []
ENABLED_ALGORITHMS = ["MOABC", "MOIABC"]

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
    "archive_size": 100,
}

MOIABC_BEST_PARAMS = {
    "tournament_size": 3,
    "elite_rate": 0.05,
    "elimination_rate": 0.10,
}
```

### 参数含义

- `RUN_TIMES`：每个测试函数的独立运行次数
- `bee`：蜂群规模，也对应部分对比算法的种群规模
- `max_iter`：最大迭代次数
- `limit`：侦察蜂重新初始化的触发阈值
- `archive_size`：多目标 Pareto 档案最大容量
- `ENABLED_SUITES`：启用哪些测试集
- `ENABLED_FUNCTION_IDS`：只运行指定测试函数；空列表表示运行已启用测试集中的全部函数
- `ENABLED_ALGORITHMS`：只运行指定算法；空列表表示运行全部算法

### 多核运行

PowerShell：

```powershell
$env:MO_SAVE_PLOTS="0"
$env:MO_SAVE_ARCHIVE_POINTS="0"
python multi_objective\run_multi_objective_comparison.py
```

并行进程数已固定为 4。

## 常用配置示例

只运行 CEC2022 的 F1 和 F6：

```python
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
```

只运行 ZDT1 和 ZDT4：

```python
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = ["ZDT1", "ZDT4"]
```

只运行部分 CEC2020 MMO 函数：

```python
ENABLED_SUITES = ["CEC2020_MMO"]
ENABLED_FUNCTION_IDS = ["MMF1", "MMF10", "MMF14"]
```

只运行指定单目标算法：

```python
ENABLED_ALGORITHMS = ["ABC", "IABC"]
```

只运行指定多目标算法：

```python
ENABLED_ALGORITHMS = ["MOABC", "MOIABC"]
```

运行基本多目标对比算法：

```python
ENABLED_ALGORITHMS = ["NSGA-II", "MOPSO"]
```

## IABC 参数敏感性分析

敏感性分析脚本位于 `single_objective/run_iabc_sensitivity.py`。

默认扫描：

```python
RUN_TIMES = 30
SEED_BASE = 20240621
ENABLED_SUITES = ["CEC2022"]

ELITE_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
ELIMINATION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
```

输出目录：

```text
single_objective/sensitivity_results/
```

主要输出文件：

- `sensitivity_detail_results.csv`：每次独立运行的详细结果
- `sensitivity_summary_by_function.csv`：每个测试函数上的参数组合统计
- `sensitivity_average_rank.csv`：参数组合平均排名

判断参数组合时优先看 `sensitivity_average_rank.csv`：

- `average_rank` 越小，整体排名越靠前
- `best_count` 越大，说明该参数组合在更多测试函数上取得第一

完整默认敏感性分析运行量较大。可以先减少 `RUN_TIMES`、测试函数数量或参数候选值做预实验。

## 输出文件

### 单目标输出

输出目录：

```text
single_objective/comparison_results/
```

主要文件：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：每次独立运行的最优值曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线
- `wilcoxon_*_vs_iabc_results.csv`：各算法与 `IABC` 的 Wilcoxon 结果
- `wilcoxon_test_results.csv`：汇总后的 Wilcoxon 结果
- `average_rank_results.csv`：跨测试函数平均排名

### 多目标输出

输出目录：

```text
multi_objective/mo_comparison_results/
```

主要文件：

- `*_results.csv`：每次独立运行的档案规模、目标和、spacing、hypervolume、耗时等
- `*_archive_points.csv`：Pareto 档案中的目标函数值
- `*_pareto_scatter.png`：Pareto 非支配解散点图
- `*_average_history.png`：平均收敛参考曲线
- `wilcoxon_*_vs_moiabc_results.csv`：各算法与 `MOIABC` 的 Wilcoxon 结果
- `wilcoxon_test_results.csv`：汇总后的 Wilcoxon 结果
- `average_rank_results.csv`：跨测试函数平均排名

## 统计说明

`wilcoxon_test_results.csv` 常用字段：

- `wins`, `ties`, `losses`：改进算法相对基准算法的胜、平、负次数
- `mean_difference`：按指标方向换算后的平均差值，正值表示改进算法更好
- `p_two_sided`：双侧检验 p 值
- `p_improved`：改进方向单侧检验 p 值
- `significant_0_05`：双侧检验是否达到 0.05 显著性水平

`average_rank_results.csv` 中的 `average_rank` 越小，表示算法在对应指标上的整体排名越靠前。

## 注意事项

- 单目标 CEC2022 当前使用项目内置的 10 维平移、旋转和打乱数据。
- 如果修改单目标实验维度，需要补充对应维度的 CEC 数据。
- `CEC2022` 单目标测试集当前为 12 个函数，即 `CEC2022_F1` 到 `CEC2022_F12`。
- `RUN_TIMES = 1` 只适合快速调试；正式实验建议使用 30 次或更多独立运行。
- `RUN_TIMES`、`bee`、`max_iter`、测试函数数量和算法数量会直接影响运行时间。
- CSV 使用 `utf-8-sig` 编码保存，便于使用 Excel 打开。
- 图像中文字体依赖系统字体，脚本默认尝试使用 `Microsoft YaHei`、`SimHei`、`SimSun`。
- `.gitignore` 已忽略结果目录、Python 缓存、`.env` 和常见密钥文件，避免误提交本地结果或凭据。
