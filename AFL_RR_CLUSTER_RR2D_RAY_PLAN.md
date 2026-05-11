# AFL RR 聚类与 RR2D 射线实验计划书

生成时间：2026-05-09
项目位置：`d:\kit`
当前相关脚本：`run_rr_afl_filter.py`

## 1. 当前节点概述

我们正在做基于 RR 间期的 AFL（房扑）候选识别。当前目标不是最终诊断，而是从 R 峰/RR 序列中筛出具有房扑样规律的片段，并输出到 `rr_afl_filter` 结果目录。

目前主线已经从最早的“相邻 RR 差值倍数关系”改成了：

```text
10min 窗口内 RR 聚类
→ 寻找整数倍 RR 模板
→ 用 allowed_rr ± 容差匹配窗口内 RR
→ 合并匹配到的 RR 生成 AFL event
```

当前已经取消了 `strong_afl / possible_afl` 分级，窗口只输出：

```text
afl / non_afl
```

最终事件仍输出：

```json
{
  "type": "af_family",
  "subtype": "flutter",
  "layer": "rr_afl_filter",
  "rule": "rr_cluster_integer_template"
}
```

## 2. 当前实现逻辑

当前 `run_rr_afl_filter.py` 的核心逻辑如下。

### 2.1 有效 RR 范围

```python
AFL_RR_MIN_MS = 300
AFL_RR_MAX_MS = 2000
```

只使用这个范围内的 RR 参与 AFL 识别。

### 2.2 RR 聚类

当前不是 KMeans，而是固定分箱 + 邻近簇合并：

```python
AFL_BIN_MS = 40
AFL_CLUSTER_MERGE_MS = 80
```

流程：

```text
RR values
→ 按 40ms 分箱
→ 每个 bin 内取 RR 平均值作为初始 cluster center
→ 相邻 cluster center 距离 <= 80ms 时合并
→ 合并后 center 用加权平均
```

也就是说，目前已经是“先归类，再计算平均值”。

### 2.3 候选簇筛选

只有出现次数或占比达到要求的簇才参与模板判断：

```python
AFL_MIN_CLUSTER_COUNT = 8
AFL_MIN_CLUSTER_RATIO = 0.03
AFL_MAX_CANDIDATE_CLUSTERS = 8
```

含义：

```text
cluster_count >= 8
或
cluster_ratio >= 3%
```

否则认为太少，直接不参与倍数模板判断。

### 2.4 整数倍模板判断

当前模板逻辑是：

```text
center ≈ k × base_rr, k = 1, 2, 3, 4
```

相关参数：

```python
AFL_TOL_MS = 50
AFL_MAX_MULTIPLE = 4
AFL_MIN_MATCHED_CLUSTERS = 2
AFL_MIN_MULTIPLIER_LEVELS = 2
AFL_MIN_SPAN_RATIO = 1.4
```

目前已经收紧过，避免正常窦律小幅波动误判：

```text
必须至少两个不同 multiplier
必须包含 k >= 2
allowed_rr 最大值 / 最小值 >= 1.4
```

例如：

```text
500, 1000 通过
760, 800, 840 不通过
```

### 2.5 RR 匹配

如果窗口发现模板：

```text
allowed_rr_ms = [500, 1000]
```

则窗口内 RR 若满足：

```text
abs(RR - allowed_rr) <= AFL_TOL_MS
```

会被计为 AFL-like RR。

### 2.6 RR2D 转移集中度

当前有一个简单 RR2D 集中度参数：

```python
AFL_MIN_TRANSITION_COVERAGE = 0.45
AFL_TRANSITION_TOP_K = 6
```

它计算相邻 RR 对：

```text
(RR_i, RR_{i+1})
```

是否集中在少数几个二维 bin 上。

注意：这个指标主要用于排除 AF 式随机散点云，对二联律/三联律不一定有效，因为二联律本身也高度规律、RR2D 也集中。

### 2.7 事件生成

当前事件合并参数：

```python
AFL_EVENT_GAP_MS = 10_000
AFL_FINAL_MIN_DURATION_MS = 60_000
```

含义：

```text
AFL-like RR 之间允许最大 10s gap
最终 event 至少 60s
```

## 3. 当前观察到的问题

经过测试，目前效果比之前好，但仍有些敏感。主要问题：

```text
房早 / 室早 / 二联律 / 三联律等规律性早搏模式会影响 AFL 判断
```

原因：

1. 早搏也会形成稳定 RR 簇；
2. 二联律/三联律本身是规律性的，不是偶发；
3. 在 RR 聚类和 RR2D 散点图上，它们可能看起来很像 AFL；
4. 如果早搏产生的 RR 水平刚好接近整数倍，例如 500/1000ms，会和 AFL RR-only 模式混淆。

当前我们暂时不打算单独实现早搏代偿规则，而是先调 AFL 参数观察效果。

## 4. 下一步优先调参方案

当前用户反馈：“效果还行，但有些过于敏感”。建议第一轮调参如下。

### 4.1 第一轮轻微变保守

建议优先改：

```python
AFL_TOL_MS = 40
AFL_MIN_MATCHED_RR_RATIO = 0.70
AFL_MIN_MINOR_COUNT = 8
AFL_MIN_MINOR_RATIO = 0.04
```

目前对应旧值：

```python
AFL_TOL_MS = 50
AFL_MIN_MATCHED_RR_RATIO = 0.60
AFL_MIN_MINOR_COUNT = 5
AFL_MIN_MINOR_RATIO = 0.03
```

含义：

- `AFL_TOL_MS`：整数倍模板和 RR 匹配容差，越小越严格；
- `AFL_MIN_MATCHED_RR_RATIO`：窗口内 RR 被模板解释的比例，越高越严格；
- `AFL_MIN_MINOR_COUNT`：模板中最小簇至少出现次数，越高越排除偶然小簇；
- `AFL_MIN_MINOR_RATIO`：模板中最小簇至少占比，越高越排除低占比小簇。

### 4.2 如果仍然敏感

可继续考虑：

```python
AFL_MIN_CLUSTER_RATIO = 0.05
AFL_MIN_SPAN_RATIO = 1.5
AFL_FINAL_MIN_DURATION_MS = 120_000
```

含义：

- 提高候选 cluster 占比门槛；
- 要求 RR 层级跨度更明显；
- 要求最终事件更长，减少短暂误报。

### 4.3 如果主要问题是事件连得太长

可考虑：

```python
AFL_EVENT_GAP_MS = 5_000
```

把 AFL-like RR 合并事件时允许的 gap 从 10s 缩到 5s。

## 5. 后续实验分支：RR2D 整数比例射线 / 格点聚类

用户提出一个重要想法：

> 如果看的是 RR_i - RR_{i+1} 散点图，AFL 或早搏等规律性模式不像普通二维聚类，而更像是从原点出发的若干条比例射线。

这是后续实验分支的核心。

### 5.1 当前 1D 方法的限制

当前 1D RR 聚类只看：

```text
RR 有哪些水平
```

例如：

```text
500ms cluster
1000ms cluster
```

它没有充分利用：

```text
一个 RR 后面跟什么 RR
```

也就是 RR 转移结构。

### 5.2 RR2D 射线想法

对相邻 RR 点：

```text
P_i = (RR_i, RR_{i+1})
```

计算比例：

```text
slope = RR_{i+1} / RR_i
```

如果 AFL 的 RR 水平是：

```text
RR ∈ {b, 2b, 3b, 4b}
```

那么 RR2D 上的点应当落在整数比例射线附近：

```text
y = (m/n) x
```

其中：

```text
m, n ∈ {1, 2, 3, 4}
```

例如：

```text
(500, 1000) -> slope = 2
(1000, 500) -> slope = 0.5
(1000, 1000) -> slope = 1
```

### 5.3 实验目标

后续可以做一个实验分支：

```text
1D RR cluster 发现候选 base
+
2D RR ratio-ray / integer-lattice 验证
+
按格点计算 mean prev RR / mean next RR
```

目标不是普通 KMeans，而是带先验结构的自定义聚类：

```text
这些点是否落在若干条整数比例射线 / 整数倍格点上？
```

### 5.4 可能的输出指标

实验分支可新增以下窗口级指标：

```text
ray_match_ratio
ray_matched_slopes
ray_transition_count
integer_lattice_match_ratio
integer_lattice_cells
mean_prev_rr_by_cell
mean_next_rr_by_cell
```

其中格点可以表示为：

```text
(k_i, k_j) 对应 (k_i × base, k_j × base)
```

例如：

```text
(1, 2): mean_prev_rr = 502, mean_next_rr = 998, count = 40
(2, 1): mean_prev_rr = 1001, mean_next_rr = 501, count = 35
(2, 2): mean_prev_rr = 998, mean_next_rr = 1003, count = 100
```

### 5.5 对早搏/室早的预期帮助

这个方法预计能减少一部分房早/室早误判，因为：

- AFL 倾向于整数倍格点：`(b, 2b)`, `(2b, b)`, `(2b, 2b)`；
- 一些早搏代偿结构如 `600 -> 1400` 的 slope 是 `2.33`，不一定落在整数比射线；
- 但如果早搏刚好形成 `500 -> 1000`，RR-only 仍可能混淆。

因此预期结论：

```text
RR2D 射线/格点聚类可减少一部分房早/室早假阳性，但不能完全替代形态学或 PVC/PAC 标注。
```

## 6. 暂不做的内容

目前暂不优先做：

```text
PVC/PAC 形态标注融合
短-长代偿间歇专门规则
KMeans 聚类
```

原因：

- KMeans 需要指定 K，容易把 AF 散点硬分成簇；
- 当前问题更适合自定义“整数比例射线/格点”聚类，而不是普通欧氏距离聚类；
- 用户目前希望先调参，再做 RR2D 射线实验分支。

## 7. 恢复当前节点时的建议指令

如果之后重新打开本计划书，可以从这里继续：

1. 先读取 `run_rr_afl_filter.py`；
2. 确认当前参数是否仍是本计划书里的版本；
3. 第一阶段先按第 4 节调参；
4. 如果调参后仍受早搏影响，再创建 RR2D ratio-ray / integer-lattice 实验分支；
5. 实验分支不要直接替换主逻辑，先输出额外 CSV 指标对比效果。

推荐后续对助手说：

```text
请根据 AFL_RR_CLUSTER_RR2D_RAY_PLAN.md 恢复上下文。我们先从第 4 节调参开始。
```
