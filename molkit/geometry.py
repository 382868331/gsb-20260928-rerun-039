"""几何分析与空间对齐。

Kabsch 算法求最优刚体变换；默认拒绝把镜像翻转当作普通旋转。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapping import MappingResult, build_mapping
from .structure import Structure, write_xyz


@dataclass
class Transform:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    per_pair_distances: np.ndarray
    mapping: MappingResult
    transformed_mobile: Structure
    reflection_detected: bool


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    a = np.asarray(coords_a, dtype=float).reshape(-1, 3)
    b = np.asarray(coords_b, dtype=float).reshape(-1, 3)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("两组坐标形状必须一致且非空")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def kabsch(
    mobile: np.ndarray,
    target: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> tuple[Transform, bool]:
    """求行约定的 R、t，使逐点 ``x @ R.T + t`` 叠加到 target。

    返回 (Transform, reflection_detected)。allow_reflection=False 时，
    若无约束最优解是镜像（行列式为 -1），改用禁止翻转的最近旋转，
    并把 reflection_detected 置 True 提醒调用者数据可能是镜像构象。
    """

    p = np.asarray(mobile, dtype=float).reshape(-1, 3)
    q = np.asarray(target, dtype=float).reshape(-1, 3)
    if p.shape != q.shape:
        raise ValueError("两组对应点形状必须一致")
    if len(p) < 3:
        raise ValueError("Kabsch 对齐至少需要 3 个对应原子")

    p_center = p.mean(axis=0)
    q_center = q.mean(axis=0)
    p_centered = p - p_center
    q_centered = q - q_center

    covariance = q_centered.T @ p_centered
    u, _, vt = np.linalg.svd(covariance)

    reflection_detected = float(np.linalg.det(u @ vt)) < 0.0
    signs = np.ones(3)
    if reflection_detected and not allow_reflection:
        signs[2] = -1.0
    rotation = u @ np.diag(signs) @ vt
    translation = q_center - p_center @ rotation.T
    return Transform(rotation=rotation, translation=translation), reflection_detected


def apply_transform(structure: Structure, transform: Transform) -> Structure:
    """返回变换后的新结构，原结构不变。"""

    moved = [
        atom.copy_with(atom.xyz @ transform.rotation.T + transform.translation)
        for atom in structure.atoms
    ]
    return Structure(atoms=moved, comment=structure.comment)


def align_structures(
    ref: Structure,
    mobile: Structure,
    explicit_pairs: list[tuple[int, int]] | None = None,
    *,
    mapping: MappingResult | None = None,
    allow_reflection: bool = False,
    require_complete: bool = False,
    strict_elements: bool = True,
) -> AlignmentResult:
    """把 mobile 空间对齐到 ref，只用 mapping 中确认的对应。

    require_complete=True 时，存在未匹配原子会报错；否则照常对齐，
    未匹配部分仍通过 AlignmentResult.mapping 返回给调用者。
    """

    if mapping is None:
        mapping = build_mapping(
            ref, mobile, explicit_pairs, strict_elements=strict_elements
        )
    if require_complete and not mapping.is_complete:
        raise ValueError(
            "存在无法匹配或对称歧义的原子: "
            f"ref={mapping.unmatched_ref} mobile={mapping.unmatched_mob}"
        )
    if len(mapping.pairs) < 3:
        raise ValueError(
            f"只有 {len(mapping.pairs)} 个对应原子，至少需要 3 个才能刚体对齐"
        )

    ref_indices = [pair[0] for pair in mapping.pairs]
    mob_indices = [pair[1] for pair in mapping.pairs]
    ref_coords = ref.coords[ref_indices]
    mob_coords = mobile.coords[mob_indices]

    transform, reflection_detected = kabsch(
        mob_coords, ref_coords, allow_reflection=allow_reflection
    )
    aligned_matched = mob_coords @ transform.rotation.T + transform.translation
    distances = np.linalg.norm(aligned_matched - ref_coords, axis=1)
    aligned_mobile = apply_transform(mobile, transform)

    return AlignmentResult(
        rotation=transform.rotation,
        translation=transform.translation,
        rmsd=rmsd(aligned_matched, ref_coords),
        per_pair_distances=distances,
        mapping=mapping,
        transformed_mobile=aligned_mobile,
        reflection_detected=reflection_detected,
    )


def save_alignment(
    result: AlignmentResult,
    ref: Structure,
    mobile: Structure,
    aligned_path: str | Path,
    report_path: str | Path | None = None,
    *,
    comment: str | None = None,
) -> None:
    """保存对齐后结构与逐原子误差报告，原始结构不受影响。"""

    write_xyz(aligned_path, result.transformed_mobile, comment=comment)
    if report_path is not None:
        save_pair_report(result, ref, mobile, report_path)


def save_pair_report(
    result: AlignmentResult,
    ref: Structure,
    mobile: Structure,
    path: str | Path,
) -> None:
    """导出 CSV：对应原子残差、缺失/多余原子、对称歧义组。"""

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "ref_index",
                "ref_element",
                "ref_serial",
                "mobile_index",
                "mobile_element",
                "mobile_serial",
                "distance",
            ]
        )
        for (ref_i, mob_i), distance in zip(
            result.mapping.pairs, result.per_pair_distances
        ):
            writer.writerow(
                [
                    "matched",
                    ref_i,
                    ref.atoms[ref_i].element,
                    ref.atoms[ref_i].serial,
                    mob_i,
                    mobile.atoms[mob_i].element,
                    mobile.atoms[mob_i].serial,
                    f"{float(distance):.10g}",
                ]
            )
        for ref_i in result.mapping.unmatched_ref:
            atom = ref.atoms[ref_i]
            writer.writerow(
                ["only_in_reference", ref_i, atom.element, atom.serial, "", "", "", ""]
            )
        for mob_i in result.mapping.unmatched_mob:
            atom = mobile.atoms[mob_i]
            writer.writerow(
                ["only_in_mobile", "", "", "", mob_i, atom.element, atom.serial, ""]
            )
