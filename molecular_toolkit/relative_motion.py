from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ensemble import ConformerComparison, compare_conformer
from .regions import RegionDisplacements


@dataclass(eq=False)
class RelativeMotionPoint:
    index: int
    conformer_name: str
    source_path: str | None
    region_a: str
    region_b: str
    reference_region: str | None
    distance: float | None
    previous_distance: float | None
    distance_change: float | None
    motion_state: str | None
    angle_deg: float | None
    angle_status: str
    distance_range: tuple[float | None, float | None] | None
    angle_range: tuple[float | None, float | None] | None
    region_a_center: list[float] | None
    region_b_center: list[float] | None
    region_a_axis: list[float] | None
    region_b_axis: list[float] | None
    region_a_translation_distance: float | None
    region_b_translation_distance: float | None
    region_a_rotation_angle: float | None
    region_b_rotation_angle: float | None
    region_a_internal_rmsd: float | None
    region_b_internal_rmsd: float | None
    atom_pairs: dict[str, list[list[int]]]
    candidate_id: str | None
    correspondence_status: str
    available_interpretations: int
    decision_id: str | None = None
    alternative_measurements: list[dict[str, Any]] = field(default_factory=list)
    region_a_expected_atoms: int | None = None
    region_a_observed_atoms: int | None = None
    region_b_expected_atoms: int | None = None
    region_b_observed_atoms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "conformer_name": self.conformer_name,
            "source_path": self.source_path,
            "regions": [self.region_a, self.region_b],
            "reference_region": self.reference_region,
            "distance": self.distance,
            "previous_distance": self.previous_distance,
            "distance_change": self.distance_change,
            "motion_state": self.motion_state,
            "angle_deg": self.angle_deg,
            "angle_status": self.angle_status,
            "distance_range": list(self.distance_range) if self.distance_range is not None else None,
            "angle_range": list(self.angle_range) if self.angle_range is not None else None,
            "region_a_center": self.region_a_center,
            "region_b_center": self.region_b_center,
            "region_a_axis": self.region_a_axis,
            "region_b_axis": self.region_b_axis,
            "region_a_translation_distance": self.region_a_translation_distance,
            "region_b_translation_distance": self.region_b_translation_distance,
            "region_a_rotation_angle": self.region_a_rotation_angle,
            "region_b_rotation_angle": self.region_b_rotation_angle,
            "region_a_internal_rmsd": self.region_a_internal_rmsd,
            "region_b_internal_rmsd": self.region_b_internal_rmsd,
            "atom_pairs": self.atom_pairs,
            "atom_coverage": {
                self.region_a: {
                    "expected_atoms": self.region_a_expected_atoms,
                    "observed_atoms": self.region_a_observed_atoms,
                    "missing_atoms": None
                    if self.region_a_expected_atoms is None
                    else self.region_a_expected_atoms - (self.region_a_observed_atoms or 0),
                },
                self.region_b: {
                    "expected_atoms": self.region_b_expected_atoms,
                    "observed_atoms": self.region_b_observed_atoms,
                    "missing_atoms": None
                    if self.region_b_expected_atoms is None
                    else self.region_b_expected_atoms - (self.region_b_observed_atoms or 0),
                },
            },
            "candidate_id": self.candidate_id,
            "correspondence_status": self.correspondence_status,
            "available_interpretations": self.available_interpretations,
            "decision_id": self.decision_id,
            "alternative_measurements": self.alternative_measurements,
        }


@dataclass(eq=False)
class RelativeMotionSeries:
    region_a: str
    region_b: str
    reference_region: str | None
    points: list[RelativeMotionPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "molecular-toolkit-relative-motion-series",
            "version": 1,
            "region_a": self.region_a,
            "region_b": self.region_b,
            "reference_region": self.reference_region,
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
        }


def save_relative_motion_report(series: RelativeMotionSeries, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(series.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analyze_relative_motion(
    comparisons: Sequence[ConformerComparison],
    region_a: str,
    region_b: str,
    *,
    reference_region: str | None = None,
    reference=None,
    stable_reference_keys: Sequence[int] | None = None,
    region_definitions: dict[str, tuple[Sequence[int], Sequence[int] | None]] | None = None,
    decision_by_source: dict[str, str] | None = None,
) -> RelativeMotionSeries:
    points: list[RelativeMotionPoint] = []
    previous_distance: float | None = None
    decisions = decision_by_source or {}

    for index, comparison in enumerate(comparisons):
        selected = _measurement(comparison, region_a, region_b)
        measurements = [selected]
        alternatives = comparison.alternative_candidates or []
        if (
            reference is not None
            and stable_reference_keys is not None
            and region_definitions is not None
            and len(alternatives) > 1
        ):
            for candidate in alternatives:
                if candidate.candidate_id == comparison.candidate_id:
                    continue
                alternative = compare_conformer(
                    reference,
                    comparison.conformer,
                    candidate,
                    stable_reference_keys,
                    region_definitions,
                    alternative_candidates=alternatives,
                )
                alternative_measurement = _measurement(alternative, region_a, region_b)
                alternative_measurement["candidate_id"] = alternative.candidate_id
                alternative_measurement["correspondence_status"] = alternative.correspondence_status
                measurements.append(alternative_measurement)
        selected["candidate_id"] = comparison.candidate_id
        selected["correspondence_status"] = comparison.correspondence_status

        distance = selected["distance"]
        change = None if distance is None or previous_distance is None else distance - previous_distance
        source = comparison.conformer.source_path
        def expected_count(name: str) -> int | None:
            if region_definitions is None or name not in region_definitions:
                return None
            return len(region_definitions[name][0])

        expected_a = expected_count(region_a)
        expected_b = expected_count(region_b)
        points.append(
            RelativeMotionPoint(
                index=index,
                conformer_name=comparison.conformer.name,
                source_path=source,
                region_a=region_a,
                region_b=region_b,
                reference_region=reference_region,
                distance=distance,
                previous_distance=previous_distance,
                distance_change=change,
                motion_state=_motion_state(change),
                angle_deg=selected["angle"],
                angle_status=selected["angle_status"],
                distance_range=_value_range(measurements, "distance"),
                angle_range=_value_range(measurements, "angle"),
                region_a_center=selected["centers"][region_a],
                region_b_center=selected["centers"][region_b],
                region_a_axis=selected["axes"][region_a],
                region_b_axis=selected["axes"][region_b],
                region_a_translation_distance=selected["translation"][region_a],
                region_b_translation_distance=selected["translation"][region_b],
                region_a_rotation_angle=selected["rotation"][region_a],
                region_b_rotation_angle=selected["rotation"][region_b],
                region_a_internal_rmsd=selected["internal"][region_a],
                region_b_internal_rmsd=selected["internal"][region_b],
                atom_pairs=selected["pairs"],
                candidate_id=comparison.candidate_id,
                correspondence_status=comparison.correspondence_status,
                available_interpretations=len(measurements),
                decision_id=decisions.get(source or ""),
                alternative_measurements=[_measurement_summary(item) for item in measurements],
                region_a_expected_atoms=expected_a,
                region_a_observed_atoms=len(selected["pairs"][region_a]),
                region_b_expected_atoms=expected_b,
                region_b_observed_atoms=len(selected["pairs"][region_b]),
            )
        )
        if distance is not None:
            previous_distance = distance

    return RelativeMotionSeries(region_a, region_b, reference_region, points)


def correspondence_decision_impact(point_count: int, start_index: int, end_index: int) -> dict[str, list[int]]:
    if not 0 <= start_index <= end_index < point_count:
        raise ValueError("Invalid correspondence decision range")
    affected_segments = [index for index in range(start_index, end_index + 1) if index < point_count - 1]
    if start_index > 0:
        affected_segments.insert(0, start_index - 1)
    return {
        "affected_point_indices": list(range(start_index, end_index + 1)),
        "affected_segment_indices": affected_segments,
    }


def _measurement(comparison: ConformerComparison, name_a: str, name_b: str) -> dict[str, Any]:
    if name_a not in comparison.alignment.regions or name_b not in comparison.alignment.regions:
        raise KeyError(f"Regions {name_a!r} and {name_b!r} must both be measured")
    region_a = comparison.alignment.regions[name_a]
    region_b = comparison.alignment.regions[name_b]
    aligned = comparison.alignment.aligned_mobile
    centers_a, axis_a = _center_and_axis(region_a, aligned)
    centers_b, axis_b = _center_and_axis(region_b, aligned)
    distance = None if centers_a is None or centers_b is None else float(np.linalg.norm(centers_a - centers_b))
    angle, status = _axes_angle(axis_a, axis_b, region_a, region_b)
    return {
        "distance": distance,
        "angle": angle,
        "angle_status": status,
        "centers": {name_a: _as_list(centers_a), name_b: _as_list(centers_b)},
        "axes": {name_a: _as_list(axis_a), name_b: _as_list(axis_b)},
        "translation": {name_a: region_a.translation_distance, name_b: region_b.translation_distance},
        "rotation": {name_a: region_a.local_rotation_angle, name_b: region_b.local_rotation_angle},
        "internal": {name_a: region_a.internal_rmsd, name_b: region_b.internal_rmsd},
        "pairs": {
            name_a: [[int(ref), int(mob)] for ref, mob in zip(region_a.reference_keys, region_a.mobile_keys)],
            name_b: [[int(ref), int(mob)] for ref, mob in zip(region_b.reference_keys, region_b.mobile_keys)],
        },
    }


def _measurement_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": value.get("candidate_id"),
        "correspondence_status": value.get("correspondence_status"),
        "distance": value["distance"],
        "atom_pairs": value["pairs"],
    }


def _center_and_axis(region: RegionDisplacements, aligned_structure):
    if not region.mobile_keys:
        return None, None
    points = np.vstack([aligned_structure.atom(key).xyz for key in region.mobile_keys])
    center = points.mean(axis=0)
    centered = points - center
    if len(points) < 2 or np.linalg.matrix_rank(centered, tol=1e-8) < 1:
        return center, None
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return center, vt[0]


def _axes_angle(axis_a, axis_b, region_a, region_b):
    if axis_a is None or axis_b is None:
        missing = [name for name, region in (("a", region_a), ("b", region_b)) if len(region.mobile_keys) < 2]
        return None, "missing_atoms" if missing else "degenerate_geometry"
    cosine = abs(float(np.dot(axis_a, axis_b) / (np.linalg.norm(axis_a) * np.linalg.norm(axis_b))))
    return float(np.degrees(np.arccos(min(1.0, max(0.0, cosine))))), "measured"


def _motion_state(change: float | None) -> str | None:
    if change is None:
        return None
    if abs(change) <= 1e-12:
        return "stable"
    return "approaching" if change < 0 else "separating"


def _value_range(values: Sequence[dict[str, Any]], key: str) -> tuple[float | None, float | None] | None:
    measured = [value[key] for value in values if value[key] is not None]
    if not measured:
        return None, None
    return float(min(measured)), float(max(measured))


def _as_list(value) -> list[float] | None:
    return None if value is None else [float(item) for item in np.asarray(value).tolist()]
