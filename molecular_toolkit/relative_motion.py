from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .candidates import CorrespondenceCandidate
from .correspondence import AtomCorrespondence, Pair
from .ensemble import Conformer, compare_conformer
from .regions import RegionDisplacements
from .structures import Structure


@dataclass(eq=False)
class RegionFrame:
    name: str
    center: np.ndarray | None
    axis: np.ndarray | None = None
    atom_pairs: list[Pair] = field(default_factory=list)
    reference_keys: list[int] = field(default_factory=list)
    conformer_keys: list[int] = field(default_factory=list)
    missing_reference_keys: list[int] = field(default_factory=list)
    missing_conformer_keys: list[int] = field(default_factory=list)
    status: str = "ok"
    internal_rmsd: float | None = None
    local_translation_distance: float | None = None
    local_rotation_angle: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "center": None if self.center is None else self.center.tolist(),
            "axis": None if self.axis is None else self.axis.tolist(),
            "atom_pairs": [
                {"reference_key": ref, "conformer_key": mobile}
                for ref, mobile in self.atom_pairs
            ],
            "reference_keys": self.reference_keys,
            "conformer_keys": self.conformer_keys,
            "missing_reference_keys": self.missing_reference_keys,
            "missing_conformer_keys": self.missing_conformer_keys,
            "status": self.status,
            "internal_rmsd": self.internal_rmsd,
            "local_translation_distance": self.local_translation_distance,
            "local_rotation_angle": self.local_rotation_angle,
        }


@dataclass(eq=False)
class RelativeMotionPoint:
    conformer_index: int
    conformer_name: str | None
    source_path: str | None
    candidate_id: str | None
    correspondence_status: str
    distance: float | None
    angle_deg: float | None
    approach: str | None
    distance_change: float | None
    region_a: RegionFrame
    region_b: RegionFrame
    atom_pairs: list[Pair]
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "conformer_index": self.conformer_index,
            "conformer_name": self.conformer_name,
            "source_path": self.source_path,
            "candidate_id": self.candidate_id,
            "correspondence_status": self.correspondence_status,
            "distance": self.distance,
            "angle_deg": self.angle_deg,
            "approach": self.approach,
            "distance_change": self.distance_change,
            "region_a": self.region_a.to_dict(),
            "region_b": self.region_b.to_dict(),
            "atom_pairs": [
                {"reference_key": ref, "conformer_key": mobile}
                for ref, mobile in self.atom_pairs
            ],
            "adopted_atoms": [
                {"reference_key": ref, "conformer_key": mobile}
                for ref, mobile in self.atom_pairs
            ],
            "note": self.note,
        }


@dataclass(eq=False)
class ValueInterval:
    lower: float | None
    upper: float | None
    selected: float | None
    values: list[float]
    status: str

    def to_dict(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "selected": self.selected,
            "values": list(self.values),
            "status": self.status,
    }


@dataclass(eq=False)
class CorrespondenceDecision:
    decision_id: str
    start_index: int
    end_index: int
    candidate_id: str | None = None
    locked_pairs: list[Pair] = field(default_factory=list)
    note: str | None = None
    active: bool = True
    affected_conformer_indices: list[int] = field(default_factory=list)
    previous_states: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "candidate_id": self.candidate_id,
            "locked_pairs": [
                {"reference_key": ref, "conformer_key": mobile}
                for ref, mobile in self.locked_pairs
            ],
            "note": self.note,
            "active": self.active,
            "affected_conformer_indices": self.affected_conformer_indices,
            "previous_states": self.previous_states,
        }


def decision_from_dict(data: dict) -> CorrespondenceDecision:
    return CorrespondenceDecision(
        decision_id=data["decision_id"],
        start_index=int(data["start_index"]),
        end_index=int(data["end_index"]),
        candidate_id=data.get("candidate_id"),
        locked_pairs=[
            (int(item["reference_key"]), int(item["conformer_key"]))
            for item in data.get("locked_pairs", [])
        ],
        note=data.get("note"),
        active=bool(data.get("active", True)),
        affected_conformer_indices=[int(i) for i in data.get("affected_conformer_indices", [])],
        previous_states=copy.deepcopy(data.get("previous_states", {})),
    )


def save_relative_motion_series(series: RelativeMotionSeries, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(series.to_dict(), indent=2, ensure_ascii=False)
    path.write_text(text + chr(10), encoding="utf-8")


@dataclass(eq=False)
class RelativeMotionSeries:
    region_a: str
    region_b: str
    reference_region: str
    points: list[RelativeMotionPoint]
    alternative_points_by_conformer: list[list[RelativeMotionPoint]] = field(default_factory=list)

    def distance_intervals(self) -> list[ValueInterval | None]:
        return [self._interval(index, "distance") for index in range(len(self.points))]

    def angle_intervals(self) -> list[ValueInterval | None]:
        return [self._interval(index, 'angle_deg') for index in range(len(self.points))]

    def trace_point(self, index: int) -> dict:
        point = self.points[index]
        return {
            "conformer_index": point.conformer_index,
            "conformer_name": point.conformer_name,
            "source_path": point.source_path,
            "candidate_id": point.candidate_id,
            "correspondence_status": point.correspondence_status,
            "adopted_atoms": point.to_dict()["atom_pairs"],
            "region_atoms": {
                self.region_a: point.region_a.to_dict()["atom_pairs"],
                self.region_b: point.region_b.to_dict()["atom_pairs"],
            },
            "decision": point.note,
        }

    def to_dict(self) -> dict:
        return {
            "format": "relative-motion-series",
            "version": 1,
            "region_a": self.region_a,
            "region_b": self.region_b,
            "reference_region": self.reference_region,
            "points": [point.to_dict() for point in self.points],
            "distance_intervals": [item.to_dict() if item is not None else None for item in self.distance_intervals()],
            "angle_intervals": [item.to_dict() if item is not None else None for item in self.angle_intervals()],
            "alternatives": [
                [point.to_dict() for point in alternatives]
                for alternatives in self.alternative_points_by_conformer
            ],
        }

    def _interval(self, index: int, attr: str) -> ValueInterval | None:
        values = [getattr(self.points[index], attr)]
        if index < len(self.alternative_points_by_conformer):
            values.extend(getattr(point, attr) for point in self.alternative_points_by_conformer[index])
        numeric = [float(value) for value in values if value is not None]
        selected = getattr(self.points[index], attr)
        if not numeric:
            return ValueInterval(None, None, selected, [], "undetermined")
        lower, upper = min(numeric), max(numeric)
        status = "selected" if upper - lower <= 1e-12 else "ambiguous"
        return ValueInterval(lower, upper, selected, numeric, status)


def analyze_relative_motion_series(
    reference: Structure,
    conformers: Sequence[Conformer],
    correspondences: Sequence[AtomCorrespondence | CorrespondenceCandidate],
    *,
    stable_reference_keys: Sequence[int],
    region_a: str,
    region_b: str,
    region_definitions: dict[str, tuple[Sequence[int], Sequence[int] | None]],
    alternative_candidates_by_conformer: Sequence[Sequence[CorrespondenceCandidate] | None] | None = None,
) -> RelativeMotionSeries:
    if {region_a, region_b} - set(region_definitions):
        raise ValueError("Both relative motion regions must be defined")
    points = []
    alternatives_series: list[list[RelativeMotionPoint]] = []
    alternative_groups = alternative_candidates_by_conformer or []
    previous_distance: float | None = None
    regions = {region_a, region_b}
    for index, (conformer, correspondence) in enumerate(zip(conformers, correspondences)):
        comparison = compare_conformer(
            reference,
            conformer,
            correspondence,
            stable_reference_keys,
            {name: region_definitions[name] for name in regions},
        )
        point = _point_from_comparison(
            index, conformer, comparison, previous_distance, region_a, region_b, reference
        )
        points.append(point)
        if point.distance is not None:
            previous_distance = point.distance

        alternatives = []
        for alternative in (alternative_groups[index] if index < len(alternative_groups) else None) or []:
            if _candidate_key(alternative) == point.candidate_id:
                continue
            alternative_comparison = compare_conformer(
                reference,
                conformer,
                alternative,
                stable_reference_keys,
                {name: region_definitions[name] for name in regions},
            )
            alternatives.append(
                _point_from_comparison(
                    index,
                    conformer,
                    alternative_comparison,
                    None,
                    region_a,
                    region_b,
                    reference,
                )
            )
        alternatives_series.append(alternatives)
    return RelativeMotionSeries(
        region_a=region_a,
        region_b=region_b,
        reference_region="custom_stable_atoms",
        points=points,
        alternative_points_by_conformer=alternatives_series,
    )

def _point_from_comparison(index, conformer, comparison, previous_distance, region_a_name, region_b_name, reference):
    frame_a = _frame(
        region_a_name,
        comparison.alignment.regions[region_a_name],
        reference,
        comparison.alignment.aligned_mobile,
    )
    frame_b = _frame(
        region_b_name,
        comparison.alignment.regions[region_b_name],
        reference,
        comparison.alignment.aligned_mobile,
    )
    distance_value = (
        None
        if frame_a.center is None or frame_b.center is None
        else float(np.linalg.norm(frame_b.center - frame_a.center))
    )
    angle_value = _axis_angle(frame_a.axis, frame_b.axis)
    distance_change = (
        None
        if previous_distance is None or distance_value is None
        else distance_value - previous_distance
    )
    approach = (
        None
        if distance_change is None
        else "approaching"
        if distance_change < 0
        else "receding"
        if distance_change > 0
        else "stationary"
    )
    note = comparison.correspondence.confirmation_note if comparison.correspondence.confirmed else None
    return RelativeMotionPoint(
        conformer_index=index,
        conformer_name=conformer.name,
        source_path=conformer.source_path,
        candidate_id=comparison.candidate_id,
        correspondence_status=comparison.correspondence_status,
        distance=distance_value,
        angle_deg=angle_value,
        approach=approach,
        distance_change=distance_change,
        region_a=frame_a,
        region_b=frame_b,
        atom_pairs=comparison.correspondence.pairs,
        note=note,
    )


def _frame(name: str, region: RegionDisplacements, reference: Structure, aligned: Structure) -> RegionFrame:
    if not region.reference_keys:
        return RegionFrame(
            name=name,
            center=None,
            atom_pairs=[],
            missing_reference_keys=list(region.missing_reference_keys),
            missing_conformer_keys=list(region.missing_mobile_keys),
            status="no_available_atoms",
            internal_rmsd=region.internal_rmsd,
            local_translation_distance=region.translation_distance,
            local_rotation_angle=region.local_rotation_angle,
        )
    points = np.vstack([aligned.atom(key).xyz for key in region.mobile_keys])
    center = points.mean(axis=0)
    axis = _directed_axis(reference, region, points, aligned)
    return RegionFrame(
        name=name,
        center=center,
        axis=axis,
        atom_pairs=list(zip(region.reference_keys, region.mobile_keys)),
        reference_keys=list(region.reference_keys),
        conformer_keys=list(region.mobile_keys),
        missing_reference_keys=list(region.missing_reference_keys),
        missing_conformer_keys=list(region.missing_mobile_keys),
        status=region.rigid_fit_status,
        internal_rmsd=region.internal_rmsd,
        local_translation_distance=region.translation_distance,
        local_rotation_angle=region.local_rotation_angle,
    )


def _directed_axis(
    reference: Structure,
    region: RegionDisplacements,
    points: np.ndarray,
    aligned: Structure,
) -> np.ndarray | None:
    if len(points) == 1:
        return None
    if len(points) == 2:
        ordered = np.vstack(
            [aligned.atom(key).xyz for key in dict(sorted(zip(region.reference_keys, region.mobile_keys))).values()]
        )
        axis = ordered[1] - ordered[0]
        norm = np.linalg.norm(axis)
        return None if norm <= 1e-12 else axis / norm
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-8) < 2:
        return None
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    ref_points = np.vstack([reference.atom(key).xyz for key in region.reference_keys])
    ref_centered = ref_points - ref_points.mean(axis=0)
    if np.dot(ref_centered[0], axis) < 0:
        axis = -axis
    return axis


def _axis_angle(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    value = float(np.degrees(np.arccos(min(1.0, max(-1.0, float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))))))
    return value


def _candidate_key(candidate: CorrespondenceCandidate) -> str | None:
    return candidate.candidate_id
