from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

from .alignment import AlignmentResult, align_structures, displacement_summary
from .candidates import CorrespondenceCandidate
from .correspondence import AtomCorrespondence
from .structures import Structure, structure_fingerprint


@dataclass(eq=False)
class ConformationComparison:
    conformer_id: str
    result: AlignmentResult
    selected_regions: dict[str, list[int]]
    reference_fingerprint: str
    mobile_fingerprint: str
    reference_source_path: str | None = None
    mobile_source_path: str | None = None
    region_coverage: dict[str, dict[str, object]] = field(default_factory=dict)
    candidate_label: str | None = None
    locked_pairs: list[tuple[int, int]] = field(default_factory=list)
    is_mirror_candidate: bool = False


@dataclass(eq=False)
class EnsembleComparison:
    reference_id: str
    comparisons: list[ConformationComparison]
    aggregate_statistics: dict[str, dict[str, float | int | list[float]]]


def compare_conformation(
    reference: Structure,
    mobile: Structure,
    correspondence: AtomCorrespondence,
    stable_keys: Iterable[int],
    flexible_regions: Mapping[str, Iterable[int]] | None = None,
    conformer_id: str = "conformer",
    candidate: CorrespondenceCandidate | None = None,
    allow_reflection: bool | None = None,
) -> ConformationComparison:
    stable = [int(key) for key in stable_keys]
    regions = {
        "stable": stable,
        **{str(name): [int(key) for key in keys] for name, keys in (flexible_regions or {}).items()},
    }
    is_mirror = bool(candidate and candidate.is_mirror)
    if allow_reflection is None:
        allow_reflection = is_mirror
    result = align_structures(
        reference,
        mobile,
        correspondence,
        allow_reflection=allow_reflection,
        fit_reference_keys=stable,
        regions=regions,
    )
    selected = {name: list(keys) for name, keys in regions.items() if name != "stable"}
    coverage = region_coverage(reference, mobile, correspondence, selected)
    return ConformationComparison(
        conformer_id=conformer_id,
        result=result,
        selected_regions=selected,
        region_coverage=coverage,
        reference_fingerprint=structure_fingerprint(reference),
        mobile_fingerprint=structure_fingerprint(mobile),
        reference_source_path=reference.metadata.get("source_path"),
        mobile_source_path=mobile.metadata.get("source_path"),
        candidate_label=candidate.label if candidate else None,
        locked_pairs=list(candidate.locked_pairs) if candidate else [],
        is_mirror_candidate=is_mirror,
    )


def compare_ensemble(
    reference: Structure,
    conformers: Sequence[tuple[str, Structure, AtomCorrespondence | CorrespondenceCandidate]],
    stable_keys: Iterable[int],
    flexible_regions: Mapping[str, Iterable[int]] | None = None,
    reference_id: str = "reference",
) -> EnsembleComparison:
    comparisons = []
    for index, item in enumerate(conformers):
        conformer_id, mobile, correspondence_or_candidate = item
        if isinstance(correspondence_or_candidate, CorrespondenceCandidate):
            candidate = correspondence_or_candidate
            correspondence = candidate.correspondence
        else:
            candidate = None
            correspondence = correspondence_or_candidate
        comparisons.append(
            compare_conformation(
                reference,
                mobile,
                correspondence,
                stable_keys,
                flexible_regions,
                conformer_id=conformer_id or f"conformer_{index}",
                candidate=candidate,
            )
        )
    return EnsembleComparison(
        reference_id=reference_id,
        comparisons=comparisons,
        aggregate_statistics=aggregate_regions(comparisons),
    )


def rank_correspondence_candidates(
    reference: Structure,
    mobile: Structure,
    candidates: Sequence[CorrespondenceCandidate],
    stable_keys: Iterable[int],
    flexible_regions: Mapping[str, Iterable[int]] | None = None,
) -> list[ConformationComparison]:
    ranked = [
        compare_conformation(
            reference,
            mobile,
            candidate.correspondence,
            stable_keys,
            flexible_regions,
            conformer_id="candidate",
            candidate=candidate,
        )
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: item.result.rmsd)


def region_coverage(
    reference: Structure,
    mobile: Structure,
    correspondence: AtomCorrespondence,
    selected_regions: Mapping[str, Iterable[int]],
) -> dict[str, dict[str, object]]:
    pair_by_reference = dict(correspondence.pairs)
    selected: set[int] = set()
    coverage = {}
    for name, keys in selected_regions.items():
        region_keys = {int(key) for key in keys}
        selected.update(region_keys)
        matched = sorted(region_keys.intersection(pair_by_reference))
        missing_reference = sorted(region_keys.difference(pair_by_reference))
        coverage[name] = {
            "selected_reference_keys": sorted(region_keys),
            "matched_reference_keys": matched,
            "missing_reference_keys": missing_reference,
            "matched_mobile_keys": [pair_by_reference[key] for key in matched],
        }
    unselected = set(range(len(reference))) - selected
    matched = sorted(unselected.intersection(pair_by_reference))
    coverage["unselected"] = {
        "selected_reference_keys": sorted(unselected),
        "matched_reference_keys": matched,
        "missing_reference_keys": [],
        "matched_mobile_keys": [pair_by_reference[key] for key in matched],
    }
    coverage["mobile_only"] = {
        "count": len(correspondence.unmatched_mobile),
        "mobile_keys": list(correspondence.unmatched_mobile),
    }
    return coverage


def aggregate_regions(
    comparisons: Sequence[ConformationComparison],
) -> dict[str, dict[str, float | int | list[float]]]:
    values: dict[str, list[float]] = {}
    for comparison in comparisons:
        for region_name, stats in comparison.result.region_statistics.items():
            values.setdefault(region_name, []).extend(stats["distances"])  # type: ignore[arg-type]
    return {name: displacement_summary(distances) for name, distances in values.items()}


def save_ensemble_report(ensemble: EnsembleComparison, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_id": ensemble.reference_id,
        "aggregate_statistics": ensemble.aggregate_statistics,
        "conformations": [comparison_to_dict(item) for item in ensemble.comparisons],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def comparison_to_dict(comparison: ConformationComparison) -> dict[str, object]:
    result = comparison.result
    correspondence = result.correspondence
    return {
        "conformer_id": comparison.conformer_id,
        "candidate_label": comparison.candidate_label,
        "locked_pairs": [
            {"reference_key": ref_key, "mobile_key": mobile_key}
            for ref_key, mobile_key in comparison.locked_pairs
        ],
        "is_mirror_candidate": comparison.is_mirror_candidate,
        "fit_reference_keys": result.fit_reference_keys,
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "rmsd": result.rmsd,
        "is_reflection": result.is_reflection,
        "pairs": [{"reference_key": r, "mobile_key": m} for r, m in correspondence.pairs],
        "unmatched_reference": correspondence.unmatched_reference,
        "unmatched_mobile": correspondence.unmatched_mobile,
        "ambiguous_identities": correspondence.ambiguous_identities,
        "paired_errors": result.paired_errors,
        "region_statistics": result.region_statistics,
        "selected_regions": comparison.selected_regions,
        "region_coverage": comparison.region_coverage,
        "reference_fingerprint": comparison.reference_fingerprint,
        "mobile_fingerprint": comparison.mobile_fingerprint,
        "reference_source_path": comparison.reference_source_path,
        "mobile_source_path": comparison.mobile_source_path,
        "mobile_metadata": result.aligned_mobile.metadata,
    }
