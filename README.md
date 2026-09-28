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
