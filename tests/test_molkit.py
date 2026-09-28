"""用已知刚体变换与人工结构验证读取、对应和对齐。"""

import math

import numpy as np
import pytest

from molkit import (
    align_structures,
    build_mapping,
    kabsch,
    read_xyz,
    resolve_serial_pairs,
    rmsd,
    save_alignment,
    structure_from_coords,
    write_xyz,
)


def rotation_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ]
    )


def irregular_molecule():
    # 非平面、不对称的一组点，避免偶然对称掩盖错误
    elements = ["N", "C", "C", "O", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.10, -0.05],
            [-0.70, 1.25, 0.20],
            [2.30, 1.05, 0.30],
            [1.80, -0.75, 0.50],
            [-1.65, 1.10, -0.35],
            [-0.30, 2.10, -0.45],
            [-0.55, 1.40, 1.20],
        ]
    )
    serials = list(range(len(elements)))
    return structure_from_coords(elements, coords, serials=serials, comment="ref")


def test_xyz_roundtrip_preserves_structure(tmp_path):
    structure = irregular_molecule()
    path = tmp_path / "mol.xyz"
    write_xyz(path, structure)
    loaded = read_xyz(path)

    assert len(loaded) == len(structure)
    assert [atom.element for atom in loaded.atoms] == [
        atom.element for atom in structure.atoms
    ]
    np.testing.assert_allclose(loaded.coords, structure.coords, atol=1e-9)


def test_xyz_strict_detects_missing_line(tmp_path):
    good = tmp_path / "good.xyz"
    good.write_text(
        "3\ncomment\n"
        "N 0 0 0\n"
        "C 1.5 0 0\n"
        "O 2.5 0.5 0\n",
        encoding="utf-8",
    )
    assert len(read_xyz(good)) == 3

    broken = tmp_path / "broken.xyz"
    broken.write_text("3\ncomment\nN 0 0 0\nC 1.5 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="实际解析出"):
        read_xyz(broken)


def test_known_rotation_and_translation_is_recovered():
    ref = irregular_molecule()
    expected_r = rotation_axis_angle([0.3, -0.8, 0.5], math.radians(67.0))
    expected_t = np.array([4.2, -1.7, 2.3])
    mobile_coords = (expected_r @ ref.coords.T).T + expected_t
    # 故意打乱输入顺序，验证对应不依赖行顺序
    permutation = [7, 2, 0, 5, 3, 1, 6, 4]
    shuffled_elements = [ref.atoms[i].element for i in permutation]
    shuffled_serials = [ref.atoms[i].serial for i in permutation]
    mobile = structure_from_coords(
        shuffled_elements,
        mobile_coords[permutation],
        serials=shuffled_serials,
    )

    pairs = resolve_serial_pairs(ref, mobile)
    result = align_structures(ref, mobile, pairs)

    np.testing.assert_allclose(result.rotation, expected_r.T, atol=1e-9)
    expected_translation = ref.coords.mean(axis=0) - mobile_coords.mean(
        axis=0
    ) @ expected_r
    np.testing.assert_allclose(result.translation, expected_translation, atol=1e-9)
    assert result.rmsd == pytest.approx(0.0, abs=1e-9)
    assert result.mapping.is_complete
    assert not result.reflection_detected
    inverse_perm = np.empty(len(permutation), dtype=int)
    inverse_perm[list(permutation)] = np.arange(len(permutation))
    np.testing.assert_allclose(
        result.transformed_mobile.coords[inverse_perm], ref.coords, atol=1e-9
    )
    # 原件不变
    np.testing.assert_allclose(mobile.coords, mobile_coords[permutation], atol=1e-12)


def test_kabsch_rejects_reflection_by_default():
    # 手性四面体被镜像：不存在能叠合的正当旋转
    target = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mirrored = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )

    transform, reflection = kabsch(mirrored, target, allow_reflection=False)
    assert reflection is True
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0, abs=1e-12)
    # 镜像无法用旋转叠合：残差明显不为零
    fitted = mirrored @ transform.rotation.T + transform.translation
    assert rmsd(fitted, target) > 0.4

    loose_transform, loose_reflection = kabsch(mirrored, target, allow_reflection=True)
    assert loose_reflection is True
    assert np.linalg.det(loose_transform.rotation) == pytest.approx(-1.0, abs=1e-12)
    loose_fitted = mirrored @ loose_transform.rotation.T + loose_transform.translation
    np.testing.assert_allclose(loose_fitted, target, atol=1e-12)


def test_missing_atoms_leave_explicit_unmatched_report():
    ref = irregular_molecule()
    # 待对齐结构缺少 3 号碳（index 2）和一个氢（index 5）
    keep = [i for i in range(len(ref)) if i not in {2, 5}]
    mobile = structure_from_coords(
        [ref.atoms[i].element for i in keep],
        ref.coords[keep] + np.array([0.5, -0.2, 0.1]),
        serials=[ref.atoms[i].serial for i in keep],
    )

    pairs = resolve_serial_pairs(ref, mobile)
    result = align_structures(ref, mobile, pairs)

    assert not result.mapping.is_complete
    assert result.mapping.unmatched_ref == [2, 5]
    assert result.mapping.unmatched_mob == []
    assert len(result.mapping.pairs) == 6
    assert result.rmsd == pytest.approx(0.0, abs=1e-9)

    with pytest.raises(ValueError, match="无法匹配"):
        align_structures(ref, mobile, pairs, require_complete=True)


def test_symmetry_requires_explicit_identity():
    # 甲烷：四个氢对称等价，不能按输入顺序糊一个身份出来
    ref = structure_from_coords(
        ["C", "H", "H", "H", "H"],
        [[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        serials=[1, 2, 3, 4, 5],
    )
    r = rotation_axis_angle([1, 1, 1], math.radians(120))
    moved = (r @ ref.coords.T).T
    mobile = structure_from_coords(
        [a.element for a in ref.atoms], moved, serials=[a.serial for a in ref.atoms]
    )

    auto = build_mapping(ref, mobile)
    assert auto.pairs == [(0, 0)]
    assert len(auto.ambiguous_groups) == 1
    group = auto.ambiguous_groups[0]
    assert group.element == "H"
    assert group.ref_indices == [1, 2, 3, 4]
    assert group.mob_indices == [1, 2, 3, 4]

    explicit = resolve_serial_pairs(ref, mobile)
    result = align_structures(ref, mobile, explicit)
    assert result.mapping.is_complete
    assert result.rmsd == pytest.approx(0.0, abs=1e-9)


def test_different_element_counts_are_ambiguous_not_guessed():
    ref = structure_from_coords(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    mobile = structure_from_coords(
        ["O", "H", "H", "H"],
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0.2, 0.9, 0]],
    )
    mapping = build_mapping(ref, mobile)
    assert mapping.pairs == [(0, 0)]
    assert mapping.unmatched_ref == [1, 2]
    assert mapping.unmatched_mob == [1, 2, 3]
    assert "数量不一致" in mapping.ambiguous_groups[0].reason


def test_duplicate_serial_is_rejected():
    structure = structure_from_coords(
        ["C", "C"], [[0, 0, 0], [1, 0, 0]], serials=[7, 7]
    )
    other = structure_from_coords(["C"], [[0, 0, 0]], serials=[7])
    with pytest.raises(ValueError, match="重复原子编号"):
        resolve_serial_pairs(structure, other)


def test_explicit_pair_validation():
    ref = structure_from_coords(["C", "O"], [[0, 0, 0], [1, 0, 0]])
    mobile = structure_from_coords(["C", "O"], [[0, 0, 0], [1, 0, 0]])

    with pytest.raises(ValueError, match="元素不一致"):
        build_mapping(ref, mobile, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="被重复对应"):
        build_mapping(ref, mobile, [(0, 0), (1, 0)], strict_elements=False)


def test_save_alignment_and_report(tmp_path):
    ref = irregular_molecule()
    expected_r = rotation_axis_angle([0.0, 0.0, 1.0], math.radians(45))
    expected_t = np.array([1.0, 2.0, 0.0])
    mobile_coords = (expected_r @ ref.coords.T).T + expected_t
    mobile = structure_from_coords(
        [a.element for a in ref.atoms],
        mobile_coords,
        serials=[a.serial for a in ref.atoms],
    )

    result = align_structures(ref, mobile, resolve_serial_pairs(ref, mobile))
    aligned_path = tmp_path / "aligned.xyz"
    report_path = tmp_path / "report.csv"
    save_alignment(result, ref, mobile, aligned_path, report_path)

    reloaded = read_xyz(aligned_path)
    np.testing.assert_allclose(reloaded.coords, ref.coords, atol=1e-8)
    text = report_path.read_text(encoding="utf-8")
    assert text.startswith("status,")
    assert text.count("matched") == 8
