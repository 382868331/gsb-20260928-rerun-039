from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .correspondence import AtomCorrespondence
from .geometry import kabsch
from .structures import Structure, save_structure, structure_fingerprint, transform_structure


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
    fit_reference_keys: list[int] = field(default_factory=list)
    region_statistics: dict[str, dict[str, float | int | list[float]]] = field(default_factory=dict)
    reference_fingerprint: str = ""
    mobile_fingerprint: str = ""
    reference_source_path: str | None = None
    mobile_source_path: str | None = None


def align_structures(
    reference: Structure,
    mobile: Structure,
    correspondence: AtomCorrespondence,
    allow_reflection: bool = False,
    fit_reference_keys: Iterable[int] | None = None,
    regions: Mapping[str, Iterable[int]] | None = None,
) -> AlignmentResult:
    pairs = list(correspondence.pairs)
    if len(pairs) < 3:
        raise ValueError("At least three explicit atom pairs are required for alignment")
    if fit_reference_keys is None:
        fit_keys = [ref_key for ref_key, _ in pairs]
    else:
        fit_keys = list(fit_reference_keys)
    fit_pair_set = set(fit_keys)
    pair_by_reference = dict(pairs)
    missing_fit = sorted(fit_pair_set - pair_by_reference.keys())
    if missing_fit:
        raise ValueError(f"Stable/fit atoms are absent from correspondence: {missing_fit}")
    fit_pairs = [(ref_key, pair_by_reference[ref_key]) for ref_key in fit_keys]
    reference_points = np.vstack([reference.atom(ref_key).xyz for ref_key, _ in fit_pairs])
    mobile_points = np.vstack([mobile.atom(mobile_key).xyz for _, mobile_key in fit_pairs])

    rotation, translation, matched_rmsd, is_reflection = kabsch(
        reference_points, mobile_points, allow_reflection=allow_reflection
    )

    aligned_mobile = transform_structure(mobile, rotation, translation)
    paired_errors = []
    region_members = {
        str(region_name): {int(key) for key in keys}
        for region_name, keys in (regions or {}).items()
    }
    if region_members:
        overlaps = [
            region_name
            for region_name, keys in region_members.items()
            if any(other_name != region_name and keys & other_keys for other_name, other_keys in region_members.items())
        ]
        if overlaps:
            raise ValueError(f"Regions must not overlap: {sorted(set(overlaps))}")
    for ref_key, mobile_key in pairs:
        displacement = reference.atom(ref_key).xyz - aligned_mobile.atom(mobile_key).xyz
        region_name = "unselected"
        for candidate_region, keys in region_members.items():
            if ref_key in keys:
                region_name = candidate_region
                break
        paired_errors.append(
            {
                "reference_key": ref_key,
                "mobile_key": mobile_key,
                "distance": float(np.linalg.norm(displacement)),
                "region": region_name,
                "used_for_fit": ref_key in fit_pair_set,
            }
        )
    region_statistics = summarize_paired_errors(paired_errors)
    return AlignmentResult(
        rotation=np.asarray(rotation, dtype=float),
        translation=np.asarray(translation, dtype=float),
        rmsd=matched_rmsd,
        correspondence=correspondence,
        aligned_mobile=aligned_mobile,
        matched_rmsd=matched_rmsd,
        paired_errors=paired_errors,
        is_reflection=is_reflection,
        fit_reference_keys=fit_keys,
        region_statistics=region_statistics,
        reference_fingerprint=structure_fingerprint(reference),
        mobile_fingerprint=structure_fingerprint(mobile),
        reference_source_path=reference.metadata.get("source_path"),
        mobile_source_path=mobile.metadata.get("source_path"),
    )


def summarize_paired_errors(
    paired_errors: list[dict[str, int | float]],
) -> dict[str, dict[str, float | int | list[float]]]:
    groups: dict[str, list[dict[str, int | float]]] = {}
    for item in paired_errors:
        groups.setdefault(str(item["region"]), []).append(item)
    return {name: displacement_summary([float(item["distance"]) for item in items]) for name, items in groups.items()}


def displacement_summary(distances: Iterable[float]) -> dict[str, float | int | list[float]]:
    values = np.asarray(list(distances), dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "rmsd": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "distances": [],
        }
    return {
        "count": int(values.size),
        "rmsd": float(np.sqrt(np.mean(values**2))),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "distances": values.tolist(),
    }


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
        "fit_reference_keys": result.fit_reference_keys,
        "region_statistics": result.region_statistics,
        "reference_fingerprint": result.reference_fingerprint,
        "mobile_fingerprint": result.mobile_fingerprint,
        "reference_source_path": result.reference_source_path,
        "mobile_source_path": result.mobile_source_path,
        "notes": result.correspondence.notes,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
