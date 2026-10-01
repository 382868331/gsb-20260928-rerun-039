from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .candidates import CorrespondenceCandidate
from .correspondence import AtomCorrespondence, Pair
from .ensemble import Conformer, compare_ensemble, save_comparison_report
from .relative_motion import (
    CorrespondenceDecision,
    analyze_relative_motion_series,
    decision_from_dict,
)
from .structures import load_structure


@dataclass(eq=False)
class ConformerCorrespondenceSpec:
    source_path: str
    name: str | None = None
    pairs: list[Pair] = field(default_factory=list)
    unmatched_reference: list[int] = field(default_factory=list)
    unmatched_mobile: list[int] = field(default_factory=list)
    ambiguous_identities: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    confirmed: bool = False
    confirmation_note: str | None = None
    notes: list[str] = field(default_factory=list)
    candidate_id: str | None = None
    alternatives: list[CorrespondenceCandidate] = field(default_factory=list)

    @classmethod
    def from_correspondence(
        cls,
        source_path: str | Path,
        correspondence: AtomCorrespondence,
        *,
        name: str | None = None,
        candidate_id: str | None = None,
        alternatives: Sequence[CorrespondenceCandidate] | None = None,
    ) -> "ConformerCorrespondenceSpec":
        return cls(
            source_path=str(source_path),
            name=name,
            pairs=list(correspondence.pairs),
            unmatched_reference=list(correspondence.unmatched_reference),
            unmatched_mobile=list(correspondence.unmatched_mobile),
            ambiguous_identities=dict(correspondence.ambiguous_identities),
            confirmed=correspondence.confirmed,
            confirmation_note=correspondence.confirmation_note,
            notes=list(correspondence.notes),
            candidate_id=candidate_id,
            alternatives=list(alternatives or []),
        )

    def to_correspondence(self) -> AtomCorrespondence:
        selected = None
        if self.candidate_id is not None:
            selected = next(
                (item for item in self.alternatives if item.candidate_id == self.candidate_id),
                None,
            )
        if selected is not None:
            return selected.to_correspondence()
        return AtomCorrespondence(
            pairs=[(int(ref), int(mob)) for ref, mob in self.pairs],
            unmatched_reference=list(self.unmatched_reference),
            unmatched_mobile=list(self.unmatched_mobile),
            ambiguous_identities=dict(self.ambiguous_identities),
            notes=list(self.notes),
            confirmed=self.confirmed,
            confirmation_note=self.confirmation_note,
        )


@dataclass(eq=False)
class ComparisonProject:
    reference_path: str
    conformers: list[ConformerCorrespondenceSpec]
    reference_regions: dict[str, list[int]]
    regions: dict[str, dict[str, list[int] | None] | tuple[Sequence[int], Sequence[int] | None]] = field(
        default_factory=dict
    )
    name: str | None = None
    notes: list[str] = field(default_factory=list)
    correspondence_decisions: list[CorrespondenceDecision] = field(default_factory=list)
    relative_motion_analyses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_regions: dict[str, dict[str, list[int] | None]] = {}
        for name, value in self.regions.items():
            if isinstance(value, dict):
                normalized_regions[name] = value
            else:
                reference_keys, mobile_keys = value
                normalized_regions[name] = {
                    "reference_keys": list(reference_keys),
                    "mobile_keys": None if mobile_keys is None else list(mobile_keys),
                }
        self.regions = normalized_regions

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "molecular-toolkit-comparison-project",
            "version": 1,
            "name": self.name,
            "reference_path": self.reference_path,
            "reference_regions": {
                name: [int(key) for key in keys] for name, keys in self.reference_regions.items()
            },
            "regions": {
                name: {
                    "reference_keys": [int(key) for key in value["reference_keys"]],
                    "mobile_keys": None
                    if value.get("mobile_keys") is None
                    else [int(key) for key in value["mobile_keys"]],
                }
                for name, value in self.regions.items()
            },
            "conformer_order": [spec.source_path for spec in self.conformers],
            "conformers": [_spec_to_dict(spec) for spec in self.conformers],
            "notes": self.notes,
            "correspondence_decisions": [decision.to_dict() for decision in self.correspondence_decisions],
            "relative_motion_analyses": self.relative_motion_analyses,
        }


def save_comparison_project(project: ComparisonProject, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_comparison_project(path: str | Path) -> ComparisonProject:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "molecular-toolkit-comparison-project":
        raise ValueError("Not a molecular toolkit comparison project")
    return ComparisonProject(
        reference_path=data["reference_path"],
        conformers=[_spec_from_dict(item) for item in data["conformers"]],
        reference_regions={
            name: [int(key) for key in keys] for name, keys in data["reference_regions"].items()
        },
        regions=data.get("regions", {}),
        name=data.get("name"),
        notes=list(data.get("notes", [])),
        correspondence_decisions=[
            decision_from_dict(item) for item in data.get("correspondence_decisions", [])
        ],
        relative_motion_analyses=dict(data.get("relative_motion_analyses", {})),
    )


def run_comparison_project(
    project: ComparisonProject,
    *,
    report_directory: str | Path | None = None,
) -> dict[str, list]:
    reference = load_structure(project.reference_path)
    loaded_conformers = [Conformer.from_file(spec.source_path, spec.name) for spec in project.conformers]
    region_args = {
        name: (value["reference_keys"], value.get("mobile_keys"))
        for name, value in project.regions.items()
    }
    runs: dict[str, list] = {}
    for reference_name, stable_keys in project.reference_regions.items():
        comparisons = compare_ensemble(
            reference,
            list(zip(loaded_conformers, [spec.to_correspondence() for spec in project.conformers])),
            stable_keys,
            region_args,
            alternative_candidates_by_conformer=[spec.alternatives for spec in project.conformers],
        )
        runs[reference_name] = comparisons
        if report_directory is not None:
            directory = Path(report_directory) / reference_name
            save_comparison_report(
                comparisons,
                directory / "comparison_report.json",
                reference=reference,
                aligned_directory=directory / "aligned",
                notes=[*project.notes, f"reference_region={reference_name}"],
            )
    if report_directory is not None:
        save_reference_region_summary(runs, Path(report_directory) / "reference_region_summary.json")
    return runs


def save_reference_region_summary(runs: dict[str, list], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_region_order": list(runs),
        "conformer_order": [item.conformer.source_path for item in next(iter(runs.values()), [])],
        "effects": reference_region_differences(runs),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reference_region_differences(runs: dict[str, list]) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    effects: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    reference_names = list(runs)
    if not reference_names:
        return effects
    for conformer_index in range(len(runs[reference_names[0]])):
        source = runs[reference_names[0]][conformer_index].conformer.source_path
        effects[source] = {}
        region_names = runs[reference_names[0]][conformer_index].alignment.regions
        for region_name in region_names:
            values = {
                reference_name: _region_metrics(
                    runs[reference_name][conformer_index].alignment.regions[region_name]
                )
                for reference_name in reference_names
            }
            baseline = values[reference_names[0]]
            effects[source][region_name] = {
                reference_name: {
                    **metrics,
                    "interpretation": _interpret_differences(
                        reference_name, reference_names[0], metrics, baseline
                    ),
                    **{
                        f"delta_{key}_from_{reference_names[0]}": _difference(metrics.get(key), baseline.get(key))
                        for key in ("rmsd", "translation_distance", "rotation_angle_deg", "internal_rmsd")
                    },
                }
                for reference_name, metrics in values.items()
            }
    return effects


def _region_metrics(region) -> dict[str, float | None]:
    return {
        "rmsd": region.rmsd if region.distances else None,
        "translation_distance": region.translation_distance,
        "rotation_angle_deg": region.local_rotation_angle,
        "internal_rmsd": region.internal_rmsd,
    }


def _difference(value, baseline):
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _interpret_differences(reference_name, baseline_name, metrics, baseline):
    if reference_name == baseline_name:
        return "Baseline reference region; differences are zero."
    labels = {
        "rmsd": "total post-alignment displacement",
        "translation_distance": "local translation",
        "rotation_angle_deg": "local rotation angle",
        "internal_rmsd": "internal deformation",
    }
    deltas = {
        key: abs(metrics[key] - baseline[key])
        for key in labels
        if metrics.get(key) is not None and baseline.get(key) is not None
    }
    if not deltas:
        return "Not enough measured atoms for comparison."
    key = max(deltas, key=deltas.get)
    signed = metrics[key] - baseline[key]
    direction = "larger" if signed > 0 else "smaller"
    return (
        f"Compared with reference region {baseline_name!r}, {labels[key]} is {abs(signed):.6g} "
        f"{direction} under {reference_name!r}."
    )


def _spec_to_dict(spec: ConformerCorrespondenceSpec) -> dict[str, Any]:
    return {
        "source_path": spec.source_path,
        "name": spec.name,
        "candidate_id": spec.candidate_id,
        "correspondence": {
            "pairs": [{"reference_key": ref, "mobile_key": mob} for ref, mob in spec.pairs],
            "unmatched_reference": spec.unmatched_reference,
            "unmatched_mobile": spec.unmatched_mobile,
            "ambiguous_identities": spec.ambiguous_identities,
            "confirmed": spec.confirmed,
            "confirmation_note": spec.confirmation_note,
            "notes": spec.notes,
        },
        "alternatives": [candidate.to_dict() for candidate in spec.alternatives],
    }


def _spec_from_dict(data: dict[str, Any]) -> ConformerCorrespondenceSpec:
    correspondence = data.get("correspondence", {})
    pairs = [
        (int(item["reference_key"]), int(item["mobile_key"]))
        for item in correspondence.get("pairs", [])
    ]
    alternatives = [_candidate_from_dict(item) for item in data.get("alternatives", [])]
    return ConformerCorrespondenceSpec(
        source_path=data["source_path"],
        name=data.get("name"),
        pairs=pairs,
        unmatched_reference=[int(key) for key in correspondence.get("unmatched_reference", [])],
        unmatched_mobile=[int(key) for key in correspondence.get("unmatched_mobile", [])],
        ambiguous_identities=correspondence.get("ambiguous_identities", {}),
        confirmed=bool(correspondence.get("confirmed", False)),
        confirmation_note=correspondence.get("confirmation_note"),
        notes=list(correspondence.get("notes", [])),
        candidate_id=data.get("candidate_id"),
        alternatives=alternatives,
    )


def _candidate_from_dict(data: dict[str, Any]) -> CorrespondenceCandidate:
    return CorrespondenceCandidate(
        candidate_id=data["candidate_id"],
        pairs=[
            (int(item["reference_key"]), int(item["mobile_key"]))
            for item in data.get("pairs", [])
        ],
        locked_pairs=[
            (int(item["reference_key"]), int(item["mobile_key"]))
            for item in data.get("locked_pairs", [])
        ],
        unmatched_reference=[int(key) for key in data.get("unmatched_reference", [])],
        unmatched_mobile=[int(key) for key in data.get("unmatched_mobile", [])],
        ambiguous_identities=data.get("ambiguous_identities", {}),
        rmsd_proper=data.get("rmsd_proper"),
        rmsd_reflected=data.get("rmsd_reflected"),
        is_mirror_candidate=bool(data.get("is_mirror_candidate", False)),
        fit_error=data.get("fit_error"),
        confirmed=bool(data.get("confirmed", False)),
        confirmation_note=data.get("confirmation_note"),
    )


def confirm_sequence_correspondence(
    project: ComparisonProject,
    start_index: int,
    end_index: int,
    *,
    candidate_id: str | None = None,
    locked_pairs: Sequence[Pair] | None = None,
    note: str = "Human review confirmed this sequence correspondence.",
) -> CorrespondenceDecision:
    if not (0 <= start_index <= end_index < len(project.conformers)):
        raise IndexError("Conformer range is outside the saved sequence")
    locked = list(locked_pairs or [])
    if candidate_id is None and not locked:
        raise ValueError("Provide candidate_id or at least one locked pair")
    decision = CorrespondenceDecision(
        decision_id=str(uuid.uuid4()),
        start_index=start_index,
        end_index=end_index,
        candidate_id=candidate_id,
        locked_pairs=locked,
        note=note,
    )
    for index in range(start_index, end_index + 1):
        spec = project.conformers[index]
        selected = None
        previously_confirmed = False
        if candidate_id is not None:
            selected = next((item for item in spec.alternatives if item.candidate_id == candidate_id), None)
        decision.previous_states[str(index)] = {
            "candidate_id": spec.candidate_id,
            "pairs": list(spec.pairs),
            "unmatched_reference": list(spec.unmatched_reference),
            "unmatched_mobile": list(spec.unmatched_mobile),
            "ambiguous_identities": deepcopy(spec.ambiguous_identities),
            "notes": list(spec.notes),
            "confirmed": previously_confirmed,
            "confirmation_note": spec.confirmation_note,
        }
        if candidate_id is not None:
            if selected is None:
                raise ValueError(f"Conformer {index} has no candidate {candidate_id!r}")
            spec.candidate_id = candidate_id
            correspondence = selected.to_correspondence()
        else:
            correspondence = spec.to_correspondence()
        pair_set = set(correspondence.pairs)
        for pair in locked:
            if pair not in pair_set:
                raise ValueError(f"Locked pair {pair} is unavailable in conformer {index}")
        spec.pairs = list(correspondence.pairs)
        spec.unmatched_reference = list(correspondence.unmatched_reference)
        spec.unmatched_mobile = list(correspondence.unmatched_mobile)
        spec.ambiguous_identities = deepcopy(correspondence.ambiguous_identities)
        spec.confirmed = True
        spec.confirmation_note = note
        if "human_confirmed" not in spec.notes:
            spec.notes.append("human_confirmed")
        decision.affected_conformer_indices.append(index)
    project.correspondence_decisions.append(decision)
    return decision


def withdraw_correspondence_decision(
    project: ComparisonProject, decision_id: str
) -> CorrespondenceDecision:
    decision = next(
        (item for item in reversed(project.correspondence_decisions) if item.decision_id == decision_id),
        None,
    )
    if decision is None:
        raise KeyError(f"Unknown correspondence decision: {decision_id}")
    for index in decision.affected_conformer_indices:
        state = decision.previous_states[str(index)]
        spec = project.conformers[index]
        spec.candidate_id = state["candidate_id"]
        spec.pairs = list(state["pairs"])
        spec.unmatched_reference = list(state["unmatched_reference"])
        spec.unmatched_mobile = list(state["unmatched_mobile"])
        spec.ambiguous_identities = deepcopy(state["ambiguous_identities"])
        spec.notes = list(state["notes"])
        spec.confirmed = bool(state["confirmed"])
        spec.confirmation_note = state["confirmation_note"]
        if spec.confirmed is False and "human_confirmed" in spec.notes:
            spec.notes.remove("human_confirmed")
    decision.active = False
    return decision


def decision_affected_measurements(
    project: ComparisonProject, decision_id: str, analysis_name: str | None = None
) -> list[dict[str, Any]]:
    decision = next(item for item in project.correspondence_decisions if item.decision_id == decision_id)
    analyses = (
        {analysis_name: project.relative_motion_analyses[analysis_name]}
        if analysis_name is not None
        else project.relative_motion_analyses
    )
    affected = set(decision.affected_conformer_indices)
    results = []
    for name, saved in analyses.items():
        for point in saved.get("points", []):
            index = int(point["conformer_index"])
            if index in affected:
                results.append({"analysis": name, **point})
            elif index - 1 in affected:
                results.append({"analysis": name, "conformer_index": index, "adjacency": "change_from_affected_previous"})
    return results


def run_relative_motion_project(
    project: ComparisonProject,
    analysis_name: str,
    *,
    region_a: str,
    region_b: str,
    reference_region: str,
) -> dict[str, Any]:
    if reference_region not in project.reference_regions:
        raise ValueError(f"Unknown stable reference region: {reference_region!r}")
    reference = load_structure(project.reference_path)
    conformers = [Conformer.from_file(spec.source_path, spec.name) for spec in project.conformers]
    region_definitions = {
        name: (value["reference_keys"], value.get("mobile_keys"))
        for name, value in project.regions.items()
        if name in {region_a, region_b}
    }
    series = analyze_relative_motion_series(
        reference,
        conformers,
        [spec.to_correspondence() for spec in project.conformers],
        stable_reference_keys=project.reference_regions[reference_region],
        region_a=region_a,
        region_b=region_b,
        region_definitions=region_definitions,
        alternative_candidates_by_conformer=[spec.alternatives for spec in project.conformers],
    )
    series.reference_region = reference_region
    payload = series.to_dict()
    payload["analysis_name"] = analysis_name
    active_decisions = [decision.to_dict() for decision in project.correspondence_decisions if decision.active]
    for point in payload["points"]:
        point["active_decisions"] = [
            {
                "decision_id": item["decision_id"],
                "note": item["note"],
                "range": [item["start_index"], item["end_index"]],
            }
            for item in active_decisions
            if item["start_index"] <= point["conformer_index"] <= item["end_index"]
        ]
    project.relative_motion_analyses[analysis_name] = payload
    return payload
