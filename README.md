# GABC

本项目是人工蜂群算法（Artificial Bee Colony, ABC）及改进人工蜂群算法（GABC）的对比实验代码，包含单目标优化和多目标优化两部分。

单目标部分基于 CEC2017、CEC2022 测试函数，对比经典 ABC 与改进 GABC 的最优值、误差、耗时和收敛曲线。多目标部分基于 ZDT 系列测试函数，对比 MOABC 与 MOGABC 的 Pareto 非支配解、间距指标、超体积指标和收敛情况。

## 项目结构

```text
GABC/
|-- single_objective/
|   |-- ABC.py                         # 经典人工蜂群算法
|   |-- GABC.py                        # 改进人工蜂群算法
|   |-- compare_abc_gabc.py            # 单目标对比实验入口
|   |-- cec2017_official.py            # CEC2017 测试函数封装
|   |-- cec2022_official.py            # CEC2022 测试函数封装
|   |-- comparison_results/            # 单目标实验结果
|   |-- official_cec2017/input_data/   # CEC2017 官方数据
|   `-- official_cec2022/              # CEC2022 官方数据
|-- multi_objective/
|   |-- MOABC.py                       # 多目标人工蜂群算法
|   |-- MOGABC.py                      # 改进多目标人工蜂群算法
|   |-- compare_moabc_mogabc.py        # 多目标对比实验入口
|   |-- multiobjective_benchmarks.py   # ZDT 测试函数
|   |-- mo_utils.py                    # 多目标工具函数
|   `-- mo_comparison_results/         # 多目标实验结果
`-- README.md
```

## 环境依赖

建议使用 Python 3.9 及以上版本。项目主要依赖：

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

进入单目标目录后运行：

```bash
cd single_objective
python compare_abc_gabc.py
```

默认会运行 `CEC2017` 和 `CEC2022` 中已封装的测试函数，并比较：

- `ABC`
- `GABC`

结果会保存到：

```text
single_objective/comparison_results/
```

主要输出文件包括：

- `*_results.csv`：每次独立运行的最优值、误差、耗时和随机种子
- `*_best_value_curve.png`：独立运行最优值对比曲线
- `*_error_boxplot.png`：误差箱线图
- `*_average_convergence.png`：平均收敛曲线

## 运行多目标实验

进入多目标目录后运行：

```bash
cd multi_objective
python compare_moabc_mogabc.py
```

默认会运行 ZDT 测试函数，并比较：

- `MOABC`
- `MOGABC`

结果会保存到：

```text
multi_objective/mo_comparison_results/
```

主要输出文件包括：

- `*_results.csv`：每次独立运行的档案规模、目标和、间距指标、超体积和耗时
- `*_archive_points.csv`：Pareto 档案中的目标点
- `*_pareto_scatter.png`：Pareto 非支配解散点图
- `*_average_history.png`：平均收敛参考曲线

## 实验参数说明

单目标实验参数位于 `single_objective/compare_abc_gabc.py`：

```python
RUN_TIMES = 1
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
RUN_TIMES = 50
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = []

COMMON_PARAMS = {
    "bee": 50,
    "max_iter": 500,
    "limit": 100,
    "archive_size": 100,
}
```

其中：

- `RUN_TIMES`：独立重复运行次数
- `bee`：蜂群规模
- `max_iter`：最大迭代次数
- `limit`：侦察蜂触发阈值
- `archive_size`：多目标算法的外部档案最大容量
- `ENABLED_SUITES`：启用的测试集
- `ENABLED_FUNCTION_IDS`：指定运行的测试函数，空列表表示不过滤

例如，只运行 CEC2022 的 F1 和 F6：

```python
ENABLED_SUITES = ["CEC2022"]
ENABLED_FUNCTION_IDS = ["CEC2022_F1", "CEC2022_F6"]
```

例如，只运行 ZDT1 和 ZDT4：

```python
ENABLED_SUITES = ["ZDT"]
ENABLED_FUNCTION_IDS = ["ZDT1", "ZDT4"]
```

## 算法改进点

相较于经典 ABC，本项目中的 GABC 主要加入了以下机制：

- 佳点集初始化：提高初始种群分布均匀性
- 相对适应度与锦标赛选择：降低极端个体对选择过程的影响
- 精英增强搜索：对较优个体进行额外邻域搜索
- 末位淘汰机制：按迭代进度逐步淘汰较差个体并重新初始化

多目标 MOGABC 在此基础上结合了非支配排序、拥挤距离、外部档案维护等机制，用于获得更稳定的 Pareto 解集。

## 注意事项

- 建议从对应子目录运行实验脚本，因为脚本使用了同目录模块导入。
- 单目标 CEC2017、CEC2022 测试依赖 `official_cec2017/input_data` 和 `official_cec2022` 中的官方数据文件，不要随意删除。
- 实验运行时间与 `RUN_TIMES`、`bee`、`max_iter`、测试函数数量直接相关。
- 图表中文显示依赖系统字体，脚本默认尝试使用 `Microsoft YaHei`、`SimHei`、`SimSun`。
- CSV 文件使用 `utf-8-sig` 编码保存，便于用 Excel 打开。
