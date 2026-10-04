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

需要分析两个柔性区域之间的相对运动时，在项目中设置 `relative_motion_pairs`。运行项目后，每个参照区域目录下都会生成 `relative_motion/区域A__区域B.json`，按保存的构象顺序记录：

- 两区域中心的距离、相对前一构象的距离变化，以及 `approaching`、`separating` 或 `stable`；
- 两区域主轴夹角；原子不足或几何退化时为 `null`，并通过 `angle_status` 说明原因，不会填成零；
- 多候选对应在当前测量上的 `distance_range`、`angle_range`，保留不同解释的结果区间；
- 各区域相对当前稳定参照的平移、旋转和内部形变 RMSD，避免把更换参照造成的整体变化解释成区域内部变形；
- 每个点的构象名称、源路径、实际采用的原子对、候选 ID 和人工决定 ID。

人工可确认一个连续区间的对应关系；函数会返回受影响的点和相邻变化线段。撤回时只把该区间中仍有候选的不确定项重新打开，并追加一条撤回记录，不覆盖原决定历史：

```python
from molecular_toolkit import confirm_project_correspondence, withdraw_project_correspondence

decision = confirm_project_correspondence(
    project,
    start_index=3,
    end_index=7,
    candidate_id="cand-001",
    note="Accepted swapped-domain interpretation.",
)
decision["affected_segment_indices"]  # 包含区间边界处会改变的相邻测量
withdraw_project_correspondence(project, decision["decision_id"])
```

重开项目并重新运行时，原始坐标不会被改写；不同参照区域的结果分目录保存，人工确认、撤回、候选方案和每一点的采用原子都可从 JSON 报告追溯。

## 区域接触的形成、持续与失去

在项目中设置 `contact_rules` 后，运行结果会为每个稳定参照目录生成 `contacts/区域A__区域B__进入距离__解除距离.json`。进入和解除阈值可以不同，形成迟滞区间；阈值附近的往返不会反复制造形成/失去事件。还可设置 `min_contact_duration`，短暂靠近若未达到要求就不会生成接触事件：

```python
project.contact_rules = [
    {
        "region_a": "loop",
        "region_b": "helix",
        "enter_distance": 4.0,
        "exit_distance": 5.0,
        "min_contact_duration": 2,
    }
]
```

每个接触点记录距离、采用的候选 ID、实际原子对、构象来源、人工决定 ID 和所有候选解释。缺少关键原子时状态为 `unresolved/missing_atoms`；不同对应关系分别支持接触与不接触时为 `unresolved/interpretation_disputed`。这两种帧都不会触发接触解除，也不会把前后直接观测接触算成一段无间断接触。

区域原子全部缺失或部分缺失时都显式保留为未知：距离与主轴夹角记为 `null`，`missing_atoms` 按区域列出缺失的参照原子 key，仍配对成功的原子继续保留在 `atom_pairs` 中作为证据。缺测帧不会被当成零距离接触，也不会被当成确定分离，接触分析在缺测之后继续恢复、分离与再次接触的判定。区域一个配对原子都没有时不再抛出空数组拼接的 `ValueError`，区域报告的 `rigid_fit_status` 标记为 `missing_atoms`。

持续区间分别统计：

- `observed_contact_duration`：距离直接满足进入条件的构象数；
- `hysteresis_hold_duration`：已形成后位于进入/解除阈值之间、由迟滞规则保持的构象数；
- `confirmed_contact_duration`：带有人工决定 ID 的接触构象数；
- `unresolved_gap_duration` 与 `evidence_segments`：缺失或解释分歧造成的隔断及每段的构象范围。

区间以缺测或解释分歧帧结尾时，状态不会再误报为 `active`，而是 `unresolved`：`last_contact_index` 保留最后一次直接确认接触的构象位置，`trailing_unresolved_range`/`trailing_unresolved_duration` 给出末尾缺测范围，`end_index` 保持 `null`，同时生成一个 `unresolved` 事件。末尾未知帧既不计入接触持续，也不判定为确定分离；若后续帧恢复到迟滞保持区间，区间重新标记为 `active`，若恢复后距离达到解除阈值，则正常以 `lost` 结束。

人工确认一段候选前可先预览事件和持续时间变化，函数不修改当前项目：

```python
from molecular_toolkit import preview_project_correspondence_contact_impact

preview = preview_project_correspondence_contact_impact(
    project,
    start_index=3,
    end_index=7,
    candidate_id="cand-001",
)
```

预览中的 `before`/`after` 可直接比较接触事件、区间数量和各类持续时间。确认后再运行项目会生成正式报告；撤回沿用既有决定历史，只重新打开相关不确定构象，不覆盖原始坐标、候选或历史统计。
