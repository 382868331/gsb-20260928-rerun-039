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

除区域中心距离曲线外，还可以在项目中配置多对区域接触，并为每对分别给出进入和解除距离。距离取两区域原子间的最小距离（例如最近的一对侧链原子），每个构象都记录这对实际采用的原子，因此每个接触点、事件和持续区间都能回到具体证据：

```python
project.contact_pairs = [
    {"regions": ["left", "right"], "enter_distance": 5.0, "exit_distance": 6.0},
]
run_comparison_project(project, report_directory="reports")
# reports/contacts/left__right.json
```

### 迟滞避免阈值抖动

接触只在距离 `<= enter_distance` 时形成，只在距离 `> exit_distance` 时解除；两阈值之间保持原状态。距离在单一阈值附近往返时不会反复制造大量 `contact_formed` / `contact_lost` 事件。`enter_distance` 不允许大于 `exit_distance`。

### 缺失原子与解释分歧

- 某构象缺少接触所需原子时，该点状态为 `missing`，距离为 `null`，并在 `missing_reference_keys` 中列出缺少的区域原子；这不是接触解除。
- 多套对应候选分别支持接触与不接触时，状态为 `disputed`，`interpretations` 同时保留每套解释的候选 ID、距离、最近原子对和 `contact_support`。
- `missing` 和 `disputed` 都会发出 `contact_interrupted` 并把持续区间断开，缺失前后的两段不会被算成一个完整连续接触。
- 接触距离在每个构象自身坐标内测量，刚体参照（稳定参照区域）的改变只会产生相同的平移/旋转，不会凭空改变真实区域间接触。

### 持续区间统计与溯源

每个 episode 记录 `start_index`、`end_index`、`point_indices`、结束原因（`separated`、`missing`、`disputed`、`sequence_end`），并区分：

- `observed_frame_count`：由观测直接支持的构象数；
- `confirmed_frame_count`：人工确认对应关系后才成立的构象数。

每个点同时保存构象名称、源路径、最近原子对、两区域完整原子对、候选 ID 和决定 ID，可从任意统计回溯到坐标证据。整区域原子缺失时区域测量返回 `rigid_fit_status: missing_atoms` 而不是崩溃。

### 决定预览、确认与撤回

确认一段对应关系前可以先预览哪些接触事件和持续时间会改变，预览不修改项目：

```python
preview = preview_project_contact_decision(
    project,
    start_index=3,
    end_index=7,
    candidate_id="cand-001",
    region_pair=["left", "right"],
)
preview["left__right"]["before"]["state_sequence"]
preview["left__right"]["after"]["state_sequence"]
preview["left__right"]["events_added"]
```

确认后分歧点按所选解释判为接触/分离，备注说明此前存在分歧，事件中的 `decision_id` 标记它由人工确认成立。撤回只重新打开该区间内仍有候选的不确定项：原坐标和历史决定（含撤回记录）保持可追溯，重开后相关区间重新回到 `disputed` / `missing` 统计。
