from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .correspondence import AtomCorrespondence
from .geometry import kabsch
from .structures import Structure, save_structure, transform_structure


@dataclass(eq=False)
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    correspondence: AtomCorrespondence
    aligned_mobile: Structure
    matched_rmsd: float
    paired_errors: list[dict[str, int | float]] = field(default_factory=list)
    is_reflection: bool = False


def align_structures(
    reference: Structure,
    mobile: Structure,
    correspondence: AtomCorrespondence,
    allow_reflection: bool = False,
) -> AlignmentResult:
    pairs = correspondence.pairs
    if len(pairs) < 3:
        raise ValueError("At least three explicit atom pairs are required for alignment")
    reference_points = np.vstack([reference.atom(ref_key).xyz for ref_key, _ in pairs])
    mobile_points = np.vstack([mobile.atom(mobile_key).xyz for _, mobile_key in pairs])

    rotation, translation, matched_rmsd, is_reflection = kabsch(
        reference_points, mobile_points, allow_reflection=allow_reflection
    )

    aligned_mobile = transform_structure(mobile, rotation, translation)
    paired_errors = []
    for ref_key, mobile_key in pairs:
        displacement = reference.atom(ref_key).xyz - aligned_mobile.atom(mobile_key).xyz
        paired_errors.append(
            {
                "reference_key": ref_key,
                "mobile_key": mobile_key,
                "distance": float(np.linalg.norm(displacement)),
            }
        )
    return AlignmentResult(
        rotation=np.asarray(rotation, dtype=float),
        translation=np.asarray(translation, dtype=float),
        rmsd=matched_rmsd,
        correspondence=correspondence,
        aligned_mobile=aligned_mobile,
        matched_rmsd=matched_rmsd,
        paired_errors=paired_errors,
        is_reflection=is_reflection,
    )


def save_alignment(
    result: AlignmentResult,
    report_path: str | Path,
    aligned_structure_path: str | Path | None = None,
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if aligned_structure_path is not None:
        save_structure(result.aligned_mobile, aligned_structure_path)
    report = {
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "rmsd": result.rmsd,
        "matched_rmsd": result.matched_rmsd,
        "is_reflection": result.is_reflection,
        "pairs": [
            {"reference_key": ref_key, "mobile_key": mobile_key}
            for ref_key, mobile_key in result.correspondence.pairs
        ],
        "unmatched_reference": result.correspondence.unmatched_reference,
        "unmatched_mobile": result.correspondence.unmatched_mobile,
        "ambiguous_identities": result.correspondence.ambiguous_identities,
        "paired_errors": result.paired_errors,
        "notes": result.correspondence.notes,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
