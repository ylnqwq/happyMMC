# IABC 人工蜂群算法实验项目

本项目用于比较人工蜂群算法及相关智能优化算法在单目标优化和多目标优化问题上的表现。项目包含测试函数、算法实现、批量实验、结果统计、图像绘制和显著性检验等模块。

主要实验入口：

- 单目标入口：`single_objective/compare_abc_gabc.py`
- 多目标入口：`multi_objective/compare_moabc_mogabc.py`

单目标实验以 `IABC` 作为主要改进算法，多目标实验以 `MOIABC` 作为主要改进算法。

## 当前状态

### 单目标部分

单目标实验位于 `single_objective/`，当前比较 6 个算法：

- `ABC`：经典人工蜂群算法
- `GA`：遗传算法
- `ACO`：蚁群优化算法
- `IABC-MSS`：带多策略搜索的改进 ABC
- `NDBP-ABC`：基于邻域差分扰动的改进 ABC
- `IABC`：当前项目里的单目标主改进算法

当前默认设置：

- 独立运行次数：`RUN_TIMES = 1`
- 维度与边界：10 维，`[-100, 100]`
- 默认测试集：`CEC2017` 和 `CEC2022`
- CEC2017 函数：`CEC2017_F1`、`CEC2017_F3`、`CEC2017_F4`、`CEC2017_F5`、`CEC2017_F6`
- CEC2022 函数：`CEC2022_F1` 到 `CEC2022_F12`
- 统计指标：`best_value`、`error`，都是越小越好
- 统计比较：把 `IABC` 作为改进算法，分别与其他算法做 Wilcoxon 配对符号秩检验

注意：`RUN_TIMES = 1` 只适合快速调试，不能支持有说服力的显著性检验。正式实验建议改成 30 次或更多。

### 多目标部分

多目标实验位于 `multi_objective/`，当前比较 6 个算法：

- `MOABC`：多目标人工蜂群算法
- `NSGA-II`：非支配排序遗传算法
- `MOPSO`：多目标粒子群算法
- `Zhou-IMOABC`：Zhou 风格改进多目标 ABC
- `Zhao-IMOABC`：Zhao 风格改进多目标 ABC
- `MOIABC`：当前项目里的多目标主改进算法

当前默认设置：

- 独立运行次数：`RUN_TIMES = 10`
- 默认测试集：`CEC2020_MMO`
- 可选测试集：`ZDT`、`CEC2020_MMO`
- CEC2020 MMO 默认函数数量：24 个，包括 `MMF1`、`MMF2`、`MMF4`、`MMF5`、`MMF7`、`MMF8`、`MMF10` 到 `MMF16_L3` 等
- 统计指标：
  - `hypervolume`：越大越好
  - `spacing`：越小越好
  - `best_sum`：越小越好
- 统计比较：把 `MOIABC` 作为改进算法，分别与其他算法做 Wilcoxon 配对符号秩检验

多目标结果不是单个最优值，而是一组 Pareto 非支配解。脚本会保存 Pareto 档案、散点图、收敛参考曲线和统计结果。

## 项目结构

```text
GABC/
|-- single_objective/
|   |-- compare_abc_gabc.py              # 单目标实验入口
|   |-- single_objective_benchmarks.py   # CEC2017 / CEC2022 单目标测试函数
|   |-- embedded_cec_data.py             # 10 维 CEC 平移、旋转、打乱数据
|   |-- statistical_tests.py             # Wilcoxon 与平均排名统计
|   |-- algorithms/
|   |   |-- ABC.py
|   |   |-- GA.py
|   |   |-- ACO.py
|   |   |-- IABC.py
|   |   |-- IABC_MSS.py
|   |   `-- NDBP_ABC.py
|   `-- comparison_results/              # 单目标输出目录
|
|-- multi_objective/
|   |-- compare_moabc_mogabc.py          # 多目标实验入口
|   |-- multiobjective_benchmarks.py     # ZDT / CEC2020 MMO 测试函数
|   |-- mo_utils.py                      # Pareto 排序、拥挤距离、档案维护等工具
|   |-- statistical_tests.py             # Wilcoxon 与平均排名统计
|   |-- algorithms/
|   |   |-- MOABC.py
|   |   |-- MOIABC.py
|   |   |-- MOPSO.py
|   |   |-- NSGA2.py
|   |   |-- Zhou_IMOABC.py
|   |   `-- Zhao_IMOABC.py
|   `-- mo_comparison_results/           # 多目标输出目录
|
`-- README.md
```

## 环境依赖
本项目运行于 Python 3.12

建议使用 Python 3.9 或更高版本。

```bash
pip install numpy matplotlib
```

如果使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

项目里的 Wilcoxon 检验是自行实现的，不依赖 `scipy`。

## 运行方法

在项目根目录运行单目标实验：

```bash
python single_objective\compare_abc_gabc.py
```

在项目根目录运行多目标实验：

```bash
python multi_objective\compare_moabc_mogabc.py
```

也可以进入子目录后运行：

```bash
cd single_objective
python compare_abc_gabc.py
```

```bash
cd multi_objective
python compare_moabc_mogabc.py
```

## 参数设置

单目标参数在 `single_objective/compare_abc_gabc.py` 中修改：

```python
RUN_TIMES = 1
BOUNDS = [(-100, 100)] * 10
ENABLED_SUITES = ["CEC2017", "CEC2022"]
ENABLED_FUNCTION_IDS = []
ENABLED_ALGORITHMS = []

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
}
```

多目标参数在 `multi_objective/compare_moabc_mogabc.py` 中修改：

```python
RUN_TIMES = 10
ENABLED_SUITES = ["CEC2020_MMO"]
ENABLED_FUNCTION_IDS = []
ENABLED_ALGORITHMS = []

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
    "archive_size": 100,
}
```

参数含义：

- `RUN_TIMES`：每个测试函数的独立运行次数
- `bee`：蜂群规模，也对应部分对比算法的种群规模
- `max_iter`：最大迭代次数
- `limit`：侦察蜂重新初始化的触发阈值
- `archive_size`：多目标外部 Pareto 档案最大容量
- `ENABLED_SUITES`：启用哪些测试集
- `ENABLED_FUNCTION_IDS`：只运行指定函数；空列表表示运行已启用测试集中的全部函数
- `ENABLED_ALGORITHMS`：只运行指定算法；空列表表示运行全部算法

只运行 CEC2022 的 F1 和 F6：

```python
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
```

只运行 CEC2020 MMO 的 MMF1 和 MMF10：

```python
ENABLED_SUITES = ["CEC2020_MMO"]
ENABLED_FUNCTION_IDS = ["MMF1", "MMF10"]
```

只运行 ZDT1 和 ZDT4：

```python
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = ["ZDT1", "ZDT4"]
```

只运行指定算法：

```python
ENABLED_ALGORITHMS = ["ABC", "IABC"]          # 单目标示例
ENABLED_ALGORITHMS = ["MOABC", "MOIABC"]      # 多目标示例
ENABLED_ALGORITHMS = ["NSGA-II", "MOPSO"]     # 多目标对比算法示例
```

## 输出文件

单目标输出目录：

```text
single_objective/comparison_results/
```

主要文件：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：独立运行最优值曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线
- `wilcoxon_*_vs_iabc_results.csv`：各基准算法与 `IABC` 的 Wilcoxon 结果
- `wilcoxon_test_results.csv`：汇总后的 Wilcoxon 结果
- `average_rank_results.csv`：跨函数平均排名

多目标输出目录：

```text
multi_objective/mo_comparison_results/
```

主要文件：

- `*_results.csv`：每次独立运行的档案规模、目标和、spacing、hypervolume、耗时等
- `*_archive_points.csv`：Pareto 档案中的目标函数值
- `*_pareto_scatter.png`：Pareto 非支配解散点图
- `*_average_history.png`：平均收敛参考曲线
- `wilcoxon_*_vs_moiabc_results.csv`：各基准算法与 `MOIABC` 的 Wilcoxon 结果
- `wilcoxon_test_results.csv`：汇总后的 Wilcoxon 结果
- `average_rank_results.csv`：跨函数平均排名

## 统计说明

脚本会在所有测试函数运行完成后自动生成 Wilcoxon 配对符号秩检验和平均排名结果。

`wilcoxon_test_results.csv` 中常用字段：

- `wins`、`ties`、`losses`：改进算法相对基准算法的胜、平、负次数
- `mean_difference`：按指标方向换算后的平均差值，正值表示改进算法更好
- `p_two_sided`：双侧检验 p 值
- `p_improved`：改进方向单侧检验 p 值
- `significant_0_05`：双侧检验是否达到 0.05 显著性水平

`average_rank_results.csv` 中的 `average_rank` 越小，表示该算法在对应指标上的整体排名越靠前。

## 注意事项

- 当前单目标 CEC2017/CEC2022 默认使用项目内置的 10 维平移、旋转和打乱数据。
- 如果修改单目标实验维度，需要补充对应维度的 CEC 数据。
- 当前部分源码注释和终端中文提示存在乱码，但 Python 语法检查通过，计算逻辑可以正常加载。
- `RUN_TIMES`、`bee`、`max_iter` 和测试函数数量会直接影响运行时间。
- CSV 使用 `utf-8-sig` 编码保存，便于用 Excel 打开。
- 图像中文字体依赖系统字体，脚本默认尝试 `Microsoft YaHei`、`SimHei`、`SimSun`。
