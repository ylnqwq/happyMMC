# GABC 人工蜂群算法实验项目

本项目用于对比经典人工蜂群算法及其改进版本在单目标优化和多目标优化问题上的表现。代码分为两个部分：

- `single_objective`：单目标优化实验，对比 `ABC` 与 `GABC`
- `multi_objective`：多目标优化实验，对比 `MOABC` 与 `MOGABC`

实验基准包括 CEC2017、CEC2022 单目标测试函数，以及 ZDT、CEC2020 MMO 多目标测试函数。程序会自动完成多次独立运行、统计指标计算、CSV 结果保存和图像绘制。

## 项目结构

```text
GABC/
|-- single_objective/
|   |-- ABC.py                         # 经典人工蜂群算法
|   |-- GABC.py                        # 改进人工蜂群算法
|   |-- compare_abc_gabc.py            # 单目标对比实验入口
|   |-- cec2017_official.py            # CEC2017 测试函数封装
|   |-- cec2022_official.py            # CEC2022 测试函数封装
|   |-- comparison_results/            # 单目标实验输出目录
|   |-- official_cec2017/              # CEC2017 官方数据文件
|   `-- official_cec2022/              # CEC2022 官方数据文件
|
|-- multi_objective/
|   |-- MOABC.py                       # 多目标人工蜂群算法
|   |-- MOGABC.py                      # 改进多目标人工蜂群算法
|   |-- compare_moabc_mogabc.py        # 多目标对比实验入口
|   |-- multiobjective_benchmarks.py   # ZDT 与 CEC2020 MMO 测试函数
|   |-- mo_utils.py                    # 多目标评价与档案维护工具
|   `-- mo_comparison_results/         # 多目标实验输出目录
|
|-- 结果/                              # 实验记录与论文整理材料
`-- README.md
```

## 环境要求

建议使用 Python 3.9 及以上版本。当前实验已在 Python 3.12 环境下运行通过。

主要依赖：

```text
numpy
matplotlib
```

安装依赖：

```bash
pip install numpy matplotlib
```

如果使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

## 运行单目标实验

在项目根目录运行：

```bash
python single_objective\compare_abc_gabc.py
```

或进入子目录后运行：

```bash
cd single_objective
python compare_abc_gabc.py
```

默认对比算法：

- `ABC`：经典人工蜂群算法
- `GABC`：改进人工蜂群算法

默认测试集：

- `CEC2017`
- `CEC2022`

单目标结果保存到：

```text
single_objective/comparison_results/
```

主要输出文件：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：不同运行次数下的最优值对比曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线
- `wilcoxon_test_results.csv`：ABC 与 GABC 在每个测试函数上的 Wilcoxon 配对符号秩检验结果
- `average_rank_results.csv`：ABC 与 GABC 在全部测试函数上的平均排名汇总

## 运行多目标实验

在项目根目录运行：

```bash
python multi_objective\compare_moabc_mogabc.py
```

或进入子目录后运行：

```bash
cd multi_objective
python compare_moabc_mogabc.py
```

默认对比算法：

- `MOABC`：多目标人工蜂群算法
- `MOGABC`：改进多目标人工蜂群算法

当前多目标入口脚本默认测试集为：

```python
ENABLED_SUITES = ["CEC2020_MMO"]
```

多目标结果保存到：

```text
multi_objective/mo_comparison_results/
```

主要输出文件：

- `*_results.csv`：每次独立运行的档案规模、目标和、间距指标、超体积和耗时
- `*_archive_points.csv`：Pareto 档案中的目标函数值
- `*_pareto_scatter.png`：Pareto 非支配解散点图
- `*_average_history.png`：平均收敛参考曲线
- `wilcoxon_test_results.csv`：MOABC 与 MOGABC 在每个测试函数上的 Wilcoxon 配对符号秩检验结果
- `average_rank_results.csv`：MOABC 与 MOGABC 在全部测试函数上的平均排名汇总

## 实验参数

单目标实验参数位于 `single_objective/compare_abc_gabc.py`：

```python
RUN_TIMES = 30
BOUNDS = [(-100, 100)] * 10
ENABLED_SUITES = ["CEC2017", "CEC2022"]
ENABLED_FUNCTION_IDS = []

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
}
```

多目标实验参数位于 `multi_objective/compare_moabc_mogabc.py`：

```python
RUN_TIMES = 10
ENABLED_SUITES = ["CEC2020_MMO"]
ENABLED_FUNCTION_IDS = []

COMMON_PARAMS = {
    "bee": 75,
    "max_iter": 750,
    "limit": 150,
    "archive_size": 100,
}
```

参数含义：

- `RUN_TIMES`：每个测试函数的独立重复运行次数
- `bee`：蜂群规模
- `max_iter`：最大迭代次数
- `limit`：蜜源连续未改进达到该阈值后触发侦察蜂机制
- `archive_size`：多目标算法外部档案最大容量
- `ENABLED_SUITES`：启用的测试集
- `ENABLED_FUNCTION_IDS`：指定要运行的测试函数；空列表表示运行当前测试集中的全部函数

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

## 算法改进点

`GABC` 相比经典 `ABC` 主要加入以下机制：

- 佳点集初始化：提高初始种群分布均匀性
- 相对适应度计算：降低极端个体对选择过程的影响
- 锦标赛选择：替代传统轮盘赌选择，提高选择稳定性
- 精英增强搜索：对较优个体额外进行邻域搜索
- 末位淘汰机制：按迭代进度逐步淘汰较差个体并重新初始化

`MOGABC` 在多目标场景下进一步结合：

- 非支配排序
- 拥挤距离
- 外部 Pareto 档案维护
- 基于档案解的搜索引导
- 多目标贪婪选择策略

这些机制用于提升 Pareto 解集的收敛性、分布均匀性和稳定性。

## 多目标评价指标

多目标实验不再只有一个理论最优值，而是输出一组 Pareto 非支配解。主要评价指标如下：

- 平均非支配解数量：外部档案中保留的非支配解数量
- 平均最小目标和：档案中目标函数值之和的最小值，越小通常表示收敛性越好
- 间距指标：衡量 Pareto 解集分布均匀性，越小越均匀
- 超体积：衡量解集对目标空间的覆盖能力，越大越好
- 超体积标准差：衡量多次运行的稳定性，越小越稳定
- 平均耗时：每个算法在对应测试函数上的平均运行时间

## 显著性检验

实验脚本在全部测试函数运行完成后，会自动生成 Wilcoxon 配对符号秩检验和跨函数平均排名结果。

Wilcoxon 检验用于比较两个算法在相同随机种子、相同测试函数下的成对实验结果。输出字段中：

- `wins/ties/losses`：改进算法相对基准算法的胜、平、负次数
- `mean_difference`：按指标方向换算后的平均差值，正值表示改进算法更优
- `p_two_sided`：双侧显著性检验 p 值，通常小于 0.05 可认为差异显著
- `p_improved`：单侧检验 p 值，用于判断改进算法是否显著优于基准算法
- `average_rank`：跨测试函数平均排名，数值越小表示整体表现越好

单目标默认检验指标为 `best_value` 和 `error`，二者都是越小越好。多目标默认检验指标为 `hypervolume`、`spacing` 和 `best_sum`，其中 `hypervolume` 越大越好，`spacing` 与 `best_sum` 越小越好。

## CEC2020 MMO 实验结论摘要

根据最近一次 `CEC2020_MMO` 测试结果，`MOGABC` 在多数双目标 MMF 问题上取得了更高的平均超体积和更低的波动，尤其在 `MMF2`、`MMF10`、`MMF12`、`MMF1_E`、`MMF10_L` 和 `MMF12_L` 上优势明显，说明改进机制对复杂多峰 Pareto 前沿具有更好的收敛性和覆盖能力。

同时，`MOGABC` 的平均耗时普遍高于 `MOABC`，说明性能提升伴随一定计算开销。在部分三目标问题和 `MMF16_L` 系列上，`MOABC` 的平均超体积略优，表明当前改进策略在高维目标空间中的优势仍不完全稳定。

综合来看：

- 若关注 Pareto 前沿质量、覆盖能力和稳定性，`MOGABC` 更有优势
- 若关注计算速度，`MOABC` 更快
- 若用于论文结果分析，建议进一步加入 Wilcoxon 秩和检验或 Friedman 排名检验，以增强统计结论的严谨性

## 注意事项

- CEC2017 和 CEC2022 测试依赖 `official_cec2017`、`official_cec2022` 中的官方数据文件，不要删除这些目录。
- 实验运行时间与 `RUN_TIMES`、`bee`、`max_iter` 和测试函数数量直接相关。
- CSV 文件使用 `utf-8-sig` 编码保存，便于使用 Excel 打开。
- 图像中文字体依赖系统字体，脚本默认尝试使用 `Microsoft YaHei`、`SimHei`、`SimSun`。
- 当前部分源码注释和控制台输出存在历史编码乱码问题，但不影响核心实验逻辑和结果文件生成。
