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
result.region("loop").rmsd  # 稳定参照对齐后的总位移 RMSD
result.region("loop").translation_distance
result.region("loop").local_rotation_angle
result.region("loop").internal_rmsd  # 扣除该区域自身刚体平移/旋转后的内部形变
result.region("unselected")  # 未归入稳定集合或命名区域的对应原子
```

每个区域同时给出区域中心相对位移、局部刚体旋转矩阵、旋转角/旋转轴、逐原子内部残差向量和内部形变 RMSD。少于三个原子或几何退化时不会虚构唯一三维旋转，`rigid_fit_status` 会标记为 `underdetermined_rotation` 或 `degenerate_geometry`；已有总位移仍保留，避免把参照对齐吸收的整体运动误写成区域没有变化。

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

## 项目保存、重开与多参照区域

需要跨会话复现实验时，使用 `ComparisonProject` 保存源结构路径、构象顺序、人工确认的原子对或候选、所有可选解释、参照区域和局部区域定义：

```python
from molecular_toolkit import (
    ComparisonProject,
    ConformerCorrespondenceSpec,
    load_comparison_project,
    run_comparison_project,
    save_comparison_project,
)

project = ComparisonProject(
    reference_path="reference.xyz",
    conformers=[
        ConformerCorrespondenceSpec.from_correspondence(path, correspondence)
        for path, correspondence in zip(paths, correspondences)
    ],
    reference_regions={
        "core": [0, 1, 2, 3],
        "domain_b": [8, 9, 10, 11],
    },
    regions={"loop": ([20, 21, 22], None)},
)
save_comparison_project(project, "comparison_project.json")

reopened = load_comparison_project("comparison_project.json")
runs = run_comparison_project(reopened, report_directory="reports")
```

`run_comparison_project` 会按保存的构象顺序分别为每个参照区域重算一份报告；`reports/reference_region_summary.json` 对相同局部区域在不同参照下的总 RMSD、平移、旋转和内部形变差值进行汇总。缺失原子继续保留为未匹配；等价原子交换的所有候选及人工锁定/确认状态随项目保存，重新打开后不需要重新猜测。

## 柔性区域的相对运动序列

两个柔性区域可沿构象顺序分析相对运动。所有相对测量都先在用户选择的稳定参照原子上完成刚体对齐，再计算区域中心距离、区域轴夹角以及相对上一构象的 `approaching`、`receding` 或 `stationary` 状态；各区域自身的局部平移、旋转和内部 RMSD 同时保留，因此更换参照造成的整体变化不会被解释成区域内部变形。

```python
from molecular_toolkit import run_relative_motion_project

payload = run_relative_motion_project(
    project,
    analysis_name='hinge_opening',
    region_a='domain_a',
    region_b='domain_b',
    reference_region='stable_core',
)

payload['points'][0]['distance']
payload['distance_intervals'][0]  # 所有对应解释的距离区间，None 表示该点无法确定
payload['angle_intervals'][0]     # 无法确定角度时为 null，不会填 0
```

某些构象缺少区域原子时，只使用该构象中可用的对应原子，并记录 `missing_reference_keys`、`missing_conformer_keys` 和可用状态。存在多套原子对应时，当前人工选择作为 `selected` 值，其余候选共同给出距离和角度的结果区间；只有一个原子、轴退化或区域不可用时角度保持 `null`。

一段构象序列可统一确认对应关系，并查看该决定直接影响的测量点以及下一构象的相对变化：

```python
from molecular_toolkit import (
    confirm_sequence_correspondence,
    decision_affected_measurements,
    withdraw_correspondence_decision,
)

decision = confirm_sequence_correspondence(
    project,
    start_index=3,
    end_index=8,
    candidate_id='cand-001',
    note='Symmetric tail orientation accepted for frames 3-8.',
)
affected = decision_affected_measurements(project, decision.decision_id, 'hinge_opening')
withdraw_correspondence_decision(project, decision.decision_id)
```

撤回会把决定前保存的候选、配对、未匹配原子、歧义分组和确认备注恢复到相关构象，只重新打开这些不确定项；决定历史仍保留且标记为非活动。项目保存时同时写入相对运动分析结果，每个图点包含构象索引/来源、候选 ID、全部采用原子、两个区域实际使用的原子以及适用的人工决定。重新打开项目后可直接从图点回到具体构象和决定，原坐标、历史候选和历史选择不会被覆盖。比较不同稳定参照或不同对应方案会生成独立分析记录，互不覆盖。
