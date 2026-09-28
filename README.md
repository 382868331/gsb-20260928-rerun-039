# 分子结构研究工具包（molkit）

第一阶段：结构读取、原子身份对应、几何分析与两份结构的空间对齐。
后续的构象比较与相互作用分析复用这里建立的原子身份。

## 模块

- `molkit.structure`：`Atom` / `Structure` 数据结构，XYZ 读写（`read_xyz`、
  `write_xyz`、`structure_from_coords`）。XYZ 首行原子数与实际行数不符会报错。
- `molkit.mapping`：原子身份对应 `build_mapping`，以及按编号解析的
  `resolve_serial_pairs`。
- `molkit.geometry`：Kabsch 刚体对齐 `align_structures` / `kabsch`、`rmsd`、
  变换 `apply_transform`、结果保存 `save_alignment` / `save_pair_report`。

## 身份对应规则

- 对应关系不依赖文件中的输入顺序。
- 调用者可通过 `explicit_pairs=[(ref_index, mobile_index), ...]` 给出明确对应；
  也可让结构带原子编号，用 `resolve_serial_pairs` 解析。重复编号直接报错。
- 未显式指定的原子按元素分组：两侧各剩一个时自动配对；
  同元素多个对称等价位置、或两侧数量不一致时不猜测，
  列入 `MappingResult.ambiguous_groups` 与未匹配列表，返回给调用者。
- 默认要求显式/自动配对的元素一致；可用 `strict_elements=False` 放宽。

## 对齐

`align_structures(ref, mobile, explicit_pairs)` 返回 `AlignmentResult`：

- `rotation` / `translation`：把 mobile 叠加到 ref 的刚体变换
  （逐点 `x @ rotation.T + translation`）；
- `rmsd` 与 `per_pair_distances`：只对已确认对应的原子计算；
- `transformed_mobile`：对齐后的新结构，原始结构不被修改；
- `reflection_detected`：无约束最优解为镜像（行列式 -1）时置 True；
  默认拒绝镜像翻转，改用最近的正当旋转；`allow_reflection=True` 可放开。
- `require_complete=True` 时，存在缺失/歧义原子直接报错。

保存结果：

```python
save_alignment(result, ref, mobile, "aligned.xyz", "report.csv")
```

`report.csv` 记录每个对应对的残差，以及只在单侧出现的原子。

## 测试

```bash
.venv/bin/python -m pytest
```

覆盖：XYZ 往返与缺失行检测、已知旋转平移恢复（且输入顺序被打乱）、
手性四面体的镜像识别、缺失原子报告、对称位置必须显式对应、
元素数量不一致、重复编号、显式对应校验、结果导出。
