# GABC 人工蜂群算法实验项目

本项目实现并对比人工蜂群算法及其改进算法，包含单目标优化和多目标优化两部分。实验脚本支持多次独立运行、结果统计、图像绘制、CSV 保存和显著性检验。

## 项目结构

```text
GABC/
|-- single_objective/
|   |-- ABC.py                         # 经典人工蜂群算法
|   |-- GABC.py                        # 改进人工蜂群算法
|   |-- compare_abc_gabc.py            # 单目标实验入口
|   |-- cec2017_official.py            # CEC2017 测试函数封装
|   |-- cec2022_official.py            # CEC2022 测试函数封装
|   |-- statistical_tests.py           # 单目标统计检验工具
|   |-- comparison_results/            # 单目标实验输出目录
|   |-- official_cec2017/              # CEC2017 官方数据
|   `-- official_cec2022/              # CEC2022 官方数据
|
|-- multi_objective/
|   |-- MOABC.py                       # 多目标人工蜂群算法
|   |-- MOGABC.py                      # 改进多目标人工蜂群算法
|   |-- compare_moabc_mogabc.py        # 多目标实验入口
|   |-- multiobjective_benchmarks.py   # ZDT 与 CEC2020 MMO 测试函数
|   |-- mo_utils.py                    # 多目标工具函数
|   |-- statistical_tests.py           # 多目标统计检验工具
|   `-- mo_comparison_results/         # 多目标实验输出目录
|
|-- 结果/                              # 实验记录与整理材料
`-- README.md
```

## 环境依赖

建议使用 Python 3.9 及以上版本。

```bash
pip install numpy matplotlib
```

如果使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

## 单目标实验

单目标部分对比：

- `ABC`：经典人工蜂群算法
- `GABC`：改进人工蜂群算法

默认测试集：

- `CEC2017`
- `CEC2022`

运行命令：

```bash
python single_objective\compare_abc_gabc.py
```

也可以进入子目录运行：

```bash
cd single_objective
python compare_abc_gabc.py
```

输出目录：

```text
single_objective/comparison_results/
```

主要输出文件：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：独立运行最优值对比曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线
- `wilcoxon_test_results.csv`：ABC 与 GABC 的 Wilcoxon 配对符号秩检验结果
- `average_rank_results.csv`：ABC 与 GABC 的跨函数平均排名结果

## 多目标实验

多目标部分对比：

- `MOABC`：多目标人工蜂群算法
- `MOGABC`：改进多目标人工蜂群算法

默认测试集：

- `CEC2020_MMO`

运行命令：

```bash
python multi_objective\compare_moabc_mogabc.py
```

也可以进入子目录运行：

```bash
cd multi_objective
python compare_moabc_mogabc.py
```

输出目录：

```text
multi_objective/mo_comparison_results/
```

主要输出文件：

- `*_results.csv`：每次独立运行的档案规模、目标和、间距指标、超体积和耗时
- `*_archive_points.csv`：Pareto 档案中的目标函数值
- `*_pareto_scatter.png`：Pareto 非支配解散点图
- `*_average_history.png`：平均收敛参考曲线
- `wilcoxon_test_results.csv`：MOABC 与 MOGABC 的 Wilcoxon 配对符号秩检验结果
- `average_rank_results.csv`：MOABC 与 MOGABC 的跨函数平均排名结果

## 参数设置

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

参数说明：

- `RUN_TIMES`：每个测试函数的独立运行次数
- `bee`：蜂群规模
- `max_iter`：最大迭代次数
- `limit`：侦察蜂触发阈值
- `archive_size`：多目标外部档案最大容量
- `ENABLED_SUITES`：启用的测试集
- `ENABLED_FUNCTION_IDS`：指定运行的测试函数，空列表表示运行当前测试集中的全部函数

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

## 统计检验

实验脚本在全部测试函数运行完成后，会自动生成 Wilcoxon 配对符号秩检验和跨函数平均排名结果。

单目标默认检验指标：

- `best_value`：越小越好
- `error`：越小越好

多目标默认检验指标：

- `hypervolume`：越大越好
- `spacing`：越小越好
- `best_sum`：越小越好

`wilcoxon_test_results.csv` 中的主要字段：

- `wins`、`ties`、`losses`：改进算法相对基准算法的胜、平、负次数
- `mean_difference`：按指标方向换算后的平均差值，正值表示改进算法更优
- `p_two_sided`：双侧检验 p 值
- `p_improved`：单侧检验 p 值
- `significant_0_05`：双侧检验是否达到 0.05 显著性水平

`average_rank_results.csv` 中的 `average_rank` 越小，表示算法在对应指标上的整体排名越靠前。

## 多目标指标说明

- 非支配解数量：外部档案中保留的 Pareto 非支配解数量
- 最小目标和：档案中目标函数值之和的最小值
- 间距指标：衡量 Pareto 解集分布均匀性，越小越均匀
- 超体积：衡量 Pareto 解集对目标空间的覆盖能力，越大越好
- 平均耗时：算法在对应测试函数上的平均运行时间

## 注意事项

- CEC2017 和 CEC2022 测试依赖官方数据目录，不要删除 `official_cec2017` 和 `official_cec2022`。
- 实验运行时间与 `RUN_TIMES`、`bee`、`max_iter` 和测试函数数量直接相关。
- CSV 文件使用 `utf-8-sig` 编码保存，便于使用 Excel 打开。
- 图像中文字体依赖系统字体，脚本默认尝试使用 `Microsoft YaHei`、`SimHei`、`SimSun`。
