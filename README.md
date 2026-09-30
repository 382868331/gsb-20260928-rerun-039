# 分子结构研究工具包

当前模块只覆盖两份结构的读取、原子身份对应和刚体空间对齐，供后续构象比较与相互作用分析复用。

## 基本流程

```python
from molecular_toolkit import load_structure, match_atoms, align_structures, save_alignment

reference = load_structure("reference.pdb")
mobile = load_structure("mobile.xyz")

# serial 在两份结构中都全局唯一时可自动匹配；重复或对称位置应使用 explicit_pairs。
correspondence = match_atoms(
    reference,
    mobile,
    explicit_pairs=[(0, 0), (1, 2), (2, 5)],
)

result = align_structures(reference, mobile, correspondence)
save_alignment(result, "alignment_report.json", "aligned_mobile.xyz")
```

## 身份规则

- 每个原子有稳定的零基出现键 `key`，它只代表文件中的原子位置，不会因为原始编号重复而改变。
- `identity="serial"` 只匹配两份结构中都唯一的原始编号；缺失和重复编号会进入未匹配或歧义报告。
- `explicit_pairs` 可直接给 `(reference_key, mobile_key)`，也支持 `("serial", value)` 或 PDB 的 `(chain, residue_seq, atom_name)` 选择器。
- 工具不会根据输入顺序、元素种类或几何邻近关系自动猜测对称原子的身份。
- 对应结果返回 `pairs`、`unmatched_reference`、`unmatched_mobile` 和 `ambiguous_identities`。

## 对齐与手性

`align_structures` 使用中心化坐标和 SVD/Kabsch 求刚体变换。若最优正交矩阵的行列式为 `-1`，默认抛出 `ReflectionAlignmentError`，避免把镜像当普通旋转。只有调用者显式设置 `allow_reflection=True` 时才返回非正当变换，并在报告中标记 `is_reflection: true`。

对齐会创建新的 `aligned_mobile` 结构，不修改输入结构。报告包含旋转矩阵、平移向量、逐对原子误差、RMSD 和未匹配原子。

## 支持格式

读取和保存支持 `.xyz`、`.json`、`.pdb`/`.ent`。PDB 的五列编号可能溢出或重复，因此内部始终保留独立 `key`。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 多构象与局部柔性区域

`align_on_stable_atoms` 只使用调用者指定的稳定原子拟合刚体变换，再分别计算命名区域和未选中原子的位移分布（逐原子、RMSD、均值、中位数、95 分位、最大值），避免用整体 RMSD 掩盖局部位移：

```python
from molecular_toolkit import align_on_stable_atoms

result = align_on_stable_atoms(
    reference,
    mobile,
    correspondence.pairs,
    stable_reference_keys=[0, 1, 2, 3],
    regions={"loop": ([10, 11, 12], [10, 11, 12])},
)
result.region("loop").rmsd
result.region("unselected")  # 未归入稳定集合或命名区域的对应原子
```

每个区域还会在稳定参照系中分解为该区域自身的平移、旋转和内部形变
（`region(...).motion`）：质心平移向量与距离、局部 Kabsch 旋转的轴/角、
`rigid_rmsd`、`internal_rmsd` 和 `total_rmsd`，并满足
`total_rmsd^2 = rigid_rmsd^2 + internal_rmsd^2`。即使区域的整段平移恰好被
整体刚体拟合吸收，区域质心仍然相对于稳定参照系报告这段平移，不会误判为
“局部没有变化”。原子数不足或共线等退化情形会标记 `degenerate` 并只给平移。

结果同时返回 `stable_rmsd_proper`、`stable_rmsd_reflected` 和 `is_mirror_candidate`，用于提示镜像对应；`check_mirror` 可单独核对候选对应。除非显式 `allow_reflection=True`，镜像不会被静默接受。

## 歧义候选与人工锁定

重复 serial、对称重复结构等情况可枚举所有双射候选。缺失原子不会被补造；每个候选保留完整原子对、锁定关系、正当/镜像 RMSD：

```python
from molecular_toolkit import candidates_from_identity, lock_candidate_pairs

candidates = candidates_from_identity(reference, mobile)
best = candidates[0]
candidates = lock_candidate_pairs(candidates, [(4, 5)])  # 人工确认一对对称原子
comparison = align_on_stable_atoms(
    reference, mobile, best.pairs, stable_reference_keys=[0, 1, 2, 3]
)
```

候选数量受 `max_candidates` 限制（默认 256），歧义过大时应先锁定更多关系。

## 构象报告溯源

`Conformer.from_file` 会记录源文件路径、格式和名称。`compare_ensemble` 逐构象应用各自的对应关系，`save_comparison_report` 保存 JSON 报告，并可导出对齐后的新结构：

```python
from molecular_toolkit import Conformer, compare_ensemble, save_comparison_report

conformers = [Conformer.from_file(path) for path in paths]
comparisons = compare_ensemble(
    reference,
    list(zip(conformers, correspondences)),
    stable_reference_keys=[0, 1, 2, 3],
    regions={"loop": ([10, 11, 12], None)},
)
save_comparison_report(
    comparisons,
    "comparison_report.json",
    reference=reference,
    aligned_directory="aligned",
)
```

报告记录每个构象的来源、候选 ID、完整原子对、未匹配/歧义原子、稳定原子拟合和各区域位移。对齐只生成新坐标，原始结构对象和源文件保持不变。

## 沿构象顺序查看

`ComparisonSequence` 保持构象的输入顺序（不按 RMSD 排序），支持索引、
负索引、相邻步进而取，以及 `transitions()` 输出相邻构象间每个区域的
指标变化，方便沿轨迹顺序复查。

## 多个参照区域与差异说明

同一个区域的局部平移和旋转是在所选稳定区域定义的坐标系里测量的，换一个
参照区域结果会变；内部形变只依赖原子间相对位置，与全局刚体参照无关。
`compare_reference_frames` 对每组候选稳定原子分别重跑全部构象比较，
`explain_frame_difference` 和 `frame_summary` 并列展示各参照帧下的平移、
旋转、内部形变及其跨度，明确说明哪些差异来自参照选择、哪些是区域自身形变。

## 保存重开与可复现

`Project` 持久化人工决策和测量设置：人工对应、锁定原子对、等价原子交换的
待确认候选（`add_alternative_pairs` / `confirm_alternative`）、稳定参照区域、
各局部区域定义。保存的 JSON 记录每个源文件的 SHA-256；`Project.load` 重新
读取原始坐标文件、校验哈希并自动重算，得到一致的逐区域测量。源坐标被修改时
重开会报错而不是静默给出不可复现结果；原文件从不被改写。

```python
from molecular_toolkit import Project

project = Project.from_files("reference.xyz", ["conf1.xyz", "conf2.xyz"])
project.set_manual_pairs(0, [(0, 0), (1, 2)])       # 人工确认对称原子
project.add_alternative_pairs(0, [[(0, 0), (1, 2)]])  # 保留可选解释
project.set_stable_region([0, 1, 2, 3])
project.set_region("loop", [10, 11, 12], [10, 11, 12])
project.run()
project.save("comparison_project.json")

reopened = Project.load("comparison_project.json")  # 哈希校验后自动重算
reopened.confirm_alternative(0, 0)                 # 事后确认另一种对应
```
