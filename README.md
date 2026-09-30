# 分子结构研究工具包

模块覆盖结构读取、原子身份对应、候选关系枚举、刚体空间对齐，以及多构象的局部位移统计。

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

## 多构象与局部柔性区域

`align_structures` 可以只使用稳定原子求解变换，同时计算其他原子的误差：

```python
result = align_structures(
    reference,
    mobile,
    correspondence,
    fit_reference_keys=[0, 1, 2, 3],
    regions={"stable": [0, 1, 2, 3], "loop": [4, 5]},
)
result.region_statistics["loop"]
```

`rmsd` 始终是参与拟合的稳定原子 RMSD；`region_statistics` 分别给出稳定区、每个柔性区和未选区域的逐原子距离、RMSD、均值、中位数、范围和标准差。

多个构象可使用 `compare_ensemble` 汇总。报告为每个构象保留独立变换、误差、构象标识、输入路径元数据和坐标 SHA-256 指纹，避免不同构象或对应关系混淆。

## 候选对应和人工锁定

对称重复结构可通过 `ambiguous_groups` 枚举多种一一对应；`locked_pairs` 用于人工固定确定关系：

```python
candidates = build_correspondence_candidates(
    reference,
    mobile,
    locked_pairs=[(0, 0), (1, 1)],
    ambiguous_groups={
        "symmetric_pair": {"reference": [3, 4], "mobile": [3, 4]}
    },
    identity="none",
)
locked = lock_candidate_pairs(candidates[0], [(3, 3)], reference, mobile)
ranked = rank_correspondence_candidates(
    reference, mobile, candidates, stable_keys=[0, 1, 2, 3], flexible_regions={"loop": [4, 5]}
)
```

候选报告保留每组候选、已锁定关系、缺失原子和额外原子。镜像结构不会静默作为普通旋转处理；设置 `include_mirror=True` 会生成标记为 `is_mirror` 的候选，允许用非正当变换单独核对。

## 身份规则

- 每个原子有稳定的零基出现键 `key`，它只代表文件中的原子位置，不会因为原始编号重复而改变。
- `identity="serial"` 只匹配两份结构中都唯一的原始编号；缺失和重复编号会进入未匹配或歧义报告。
- `explicit_pairs` 可直接给 `(reference_key, mobile_key)`，也支持 `("serial", value)` 或 PDB 的 `(chain, residue_seq, atom_name)` 选择器。
- 工具不会根据输入顺序、元素种类或几何邻近关系自动猜测对称原子的身份。
- 对应结果返回 `pairs`、`unmatched_reference`、`unmatched_mobile` 和 `ambiguous_identities`；候选枚举不会丢弃人工锁定关系。

## 对齐与手性

`align_structures` 使用中心化坐标和 SVD/Kabsch 求刚体变换。若最优正交矩阵的行列式为 `-1`，默认抛出 `ReflectionAlignmentError`，避免把镜像当普通旋转。只有调用者显式设置 `allow_reflection=True` 时才返回非正当变换，并在报告中标记 `is_reflection: true`。

对齐会创建新的 `aligned_mobile` 结构，不修改输入结构。报告包含旋转矩阵、平移向量、逐对原子误差、稳定/柔性/未选区域统计、未匹配原子、构象来源和对应关系。

## 支持格式

读取和保存支持 `.xyz`、`.json`、`.pdb`/`.ent`。PDB 的五列编号可能溢出或重复，因此内部始终保留独立 `key`。

运行测试：

```bash
python -m unittest discover -s tests -v
```
