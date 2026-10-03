from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ensemble import ConformerComparison, compare_conformer


@dataclass(eq=False)
class ContactThresholds:
    """Hysteresis thresholds for a region-region contact.

    A contact forms only at or below ``enter_distance`` while separated and is
    lost only above ``exit_distance`` while in contact. Distances in the band
    between the two preserve the previous state, so values flickering around a
    single cutoff do not generate floods of events. ``enter_distance`` must not
    be larger than ``exit_distance``.
    """

    enter_distance: float
    exit_distance: float

    def __post_init__(self) -> None:
        self.enter_distance = float(self.enter_distance)
        self.exit_distance = float(self.exit_distance)
        if self.enter_distance < 0.0 or self.exit_distance < 0.0:
            raise ValueError("Contact distances must be non-negative")
        if self.enter_distance > self.exit_distance:
            raise ValueError("enter_distance must not exceed exit_distance (hysteresis band)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enter_distance": self.enter_distance,
            "exit_distance": self.exit_distance,
        }


@dataclass(eq=False)
class ContactPoint:
    index: int
    conformer_name: str
    source_path: str | None
    state: str
    distance: float | None
    closest_pair: list[int] | None
    atom_pairs: dict[str, list[list[int]]]
    missing_reference_keys: dict[str, list[int]]
    interpretations: list[dict[str, Any]]
    candidate_id: str | None
    correspondence_status: str
    decision_id: str | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "conformer_name": self.conformer_name,
            "source_path": self.source_path,
            "state": self.state,
            "distance": self.distance,
            "closest_pair": self.closest_pair,
            "atom_pairs": self.atom_pairs,
            "missing_reference_keys": self.missing_reference_keys,
            "interpretations": self.interpretations,
            "candidate_id": self.candidate_id,
            "correspondence_status": self.correspondence_status,
            "decision_id": self.decision_id,
            "note": self.note,
        }


@dataclass(eq=False)
class ContactEpisode:
    episode_id: str
    start_index: int
    end_index: int
    start_conformer: str
    end_conformer: str
    observed_frame_count: int
    confirmed_frame_count: int
    point_indices: list[int]
    end_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_conformer": self.start_conformer,
            "end_conformer": self.end_conformer,
            "frame_count": len(self.point_indices),
            "observed_frame_count": self.observed_frame_count,
            "confirmed_frame_count": self.confirmed_frame_count,
            "point_indices": list(self.point_indices),
            "end_reason": self.end_reason,
        }


@dataclass(eq=False)
class ContactTrace:
    region_a: str
    region_b: str
    thresholds: ContactThresholds
    points: list[ContactPoint] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[ContactEpisode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "molecular-toolkit-contact-persistence",
            "version": 1,
            "region_a": self.region_a,
            "region_b": self.region_b,
            "thresholds": self.thresholds.to_dict(),
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
            "events": list(self.events),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def save_contact_persistence_report(trace: ContactTrace, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analyze_contact_persistence(
    comparisons: Sequence[ConformerComparison],
    region_a: str,
    region_b: str,
    thresholds: ContactThresholds,
    *,
    reference=None,
    stable_reference_keys: Sequence[int] | None = None,
    region_definitions: dict[str, tuple[Sequence[int], Sequence[int] | None]] | None = None,
    decision_by_source: dict[str, str] | None = None,
    resolved_sources: set[str] | None = None,
) -> ContactTrace:
    """Track contact formation/persistence/loss along a conformer sequence.

    Distances are minimum atom-to-atom distances between the two regions, read
    from each conformer's own coordinates. A rigid change of alignment frame
    cannot change intra-conformer distances, so switching the stable reference
    region never invents or removes contacts. Missing atoms and interpretations
    that disagree across correspondence candidates produce ``missing`` and
    ``disputed`` states instead of being treated as loss, and they always split
    contact episodes rather than bridging the observed runs on either side.
    """
    decisions = decision_by_source or {}
    resolved = resolved_sources or set()
    trace = ContactTrace(region_a, region_b, thresholds)
    in_contact = False
    open_episode: list[int] | None = None

    for index, comparison in enumerate(comparisons):
        measurements = [
            _measurement(comparison, region_a, region_b, region_definitions)
        ]
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
                measurements.append(
                    _measurement(alternative, region_a, region_b, region_definitions)
                )

        selected = measurements[0]
        source = comparison.conformer.source_path
        interpretations = [
            {
                "candidate_id": measurement["candidate_id"],
                "distance": measurement["distance"],
                "closest_pair": measurement["closest_pair"],
                "contact_support": _contact_support(measurement["distance"], thresholds),
            }
            for measurement in measurements
        ]
        supports = {item["contact_support"] for item in interpretations}

        if selected["distance"] is None:
            state = "missing"
            note = "Required region atoms are absent in this conformer; contact cannot be judged."
        elif "contact" in supports and "no_contact" in supports and (source or "") not in resolved:
            state = "disputed"
            note = "Correspondence interpretations disagree about whether the contact exists."
        else:
            state = _hysteresis_state(selected["distance"], in_contact, thresholds)
            note = (
                "Interpretations disagreed but the correspondence was human-confirmed."
                if "contact" in supports and "no_contact" in supports
                else None
            )

        decision_id = decisions.get(source or "")
        point = ContactPoint(
            index=index,
            conformer_name=comparison.conformer.name,
            source_path=source,
            state=state,
            distance=selected["distance"],
            closest_pair=selected["closest_pair"],
            atom_pairs=selected["atom_pairs"],
            missing_reference_keys=selected["missing_reference_keys"],
            interpretations=interpretations,
            candidate_id=comparison.candidate_id,
            correspondence_status=comparison.correspondence_status,
            decision_id=decision_id,
            note=note,
        )
        trace.points.append(point)

        if state in ("missing", "disputed"):
            if open_episode is not None:
                trace.events.append(_event(point, "contact_interrupted", reason=state))
                _close_episode(trace, open_episode, trace.points, end_reason=state)
                open_episode = None
            in_contact = False
            continue

        if state == "contact":
            if not in_contact:
                trace.events.append(_event(point, "contact_formed"))
            in_contact = True
            if open_episode is None:
                open_episode = [index]
            else:
                open_episode.append(index)
        else:
            if in_contact:
                trace.events.append(_event(point, "contact_lost"))
                _close_episode(trace, open_episode or [index - 1], trace.points, end_reason="separated")
                open_episode = None
            in_contact = False

    if open_episode is not None:
        _close_episode(trace, open_episode, trace.points, end_reason="sequence_end")
    return trace


def contact_decision_preview(
    comparisons: Sequence[ConformerComparison],
    region_a: str,
    region_b: str,
    thresholds: ContactThresholds,
    start_index: int,
    end_index: int,
    candidate_id: str | None,
    *,
    reference=None,
    stable_reference_keys: Sequence[int] | None = None,
    region_definitions: dict[str, tuple[Sequence[int], Sequence[int] | None]] | None = None,
    decision_by_source: dict[str, str] | None = None,
    preview_decision_id: str = "preview-decision",
    resolved_sources: set[str] | None = None,
) -> dict[str, Any]:
    """Compare contact events/durations before and after a tentative decision.

    The project is not modified. The preview temporarily overrides the
    selected candidate (and decision attribution) over ``start_index`` to
    ``end_index`` and reports which points, events and episodes change.
    """
    if not 0 <= start_index <= end_index < len(comparisons):
        raise ValueError("Invalid contact decision range")
    overridden = []
    preview_decisions = dict(decision_by_source or {})
    preview_resolved = set(resolved_sources or set(decision_by_source or {}))
    for index in range(start_index, end_index + 1):
        comparison = comparisons[index]
        if candidate_id is not None:
            candidate = next(
                (
                    item
                    for item in (comparison.alternative_candidates or [])
                    if item.candidate_id == candidate_id
                ),
                None,
            )
            if candidate is None:
                raise ValueError(f"Conformer {index} has no candidate {candidate_id!r}")
            overridden.append(
                compare_conformer(
                    reference,
                    comparison.conformer,
                    candidate,
                    stable_reference_keys,
                    region_definitions,
                    alternative_candidates=comparison.alternative_candidates,
                )
            )
        else:
            overridden.append(comparison)
        source = comparison.conformer.source_path
        if source is not None:
            preview_decisions[source] = preview_decision_id
            preview_resolved.add(source)

    preview_comparisons = list(comparisons)
    for offset, replacement in enumerate(overridden):
        preview_comparisons[start_index + offset] = replacement

    before = analyze_contact_persistence(
        comparisons,
        region_a,
        region_b,
        thresholds,
        reference=reference,
        stable_reference_keys=stable_reference_keys,
        region_definitions=region_definitions,
        decision_by_source=decision_by_source,
        resolved_sources=resolved_sources,
    )
    after = analyze_contact_persistence(
        preview_comparisons,
        region_a,
        region_b,
        thresholds,
        reference=reference,
        stable_reference_keys=stable_reference_keys,
        region_definitions=region_definitions,
        decision_by_source=preview_decisions,
        resolved_sources=preview_resolved,
    )
    return {
        "affected_point_indices": list(range(start_index, end_index + 1)),
        "before": _trace_summary(before),
        "after": _trace_summary(after),
        "changed_point_indices": [
            index
            for index, (left, right) in enumerate(zip(before.points, after.points))
            if left.state != right.state
            or (left.distance or None) != (right.distance or None)
            or left.candidate_id != right.candidate_id
        ],
        "events_added": [
            event for event in after.events if event not in before.events
        ],
        "events_removed": [
            event for event in before.events if event not in after.events
        ],
        "episodes_before": [episode.to_dict() for episode in before.episodes],
        "episodes_after": [episode.to_dict() for episode in after.episodes],
        "preview_trace": after,
    }


def _trace_summary(trace: ContactTrace) -> dict[str, Any]:
    return {
        "state_sequence": [point.state for point in trace.points],
        "events": list(trace.events),
        "episode_count": len(trace.episodes),
        "total_contact_frames": sum(len(episode.point_indices) for episode in trace.episodes),
        "observed_contact_frames": sum(episode.observed_frame_count for episode in trace.episodes),
        "confirmed_contact_frames": sum(episode.confirmed_frame_count for episode in trace.episodes),
    }


def _hysteresis_state(distance: float, in_contact: bool, thresholds: ContactThresholds) -> str:
    if in_contact:
        return "separated" if distance > thresholds.exit_distance else "contact"
    return "contact" if distance <= thresholds.enter_distance else "separated"


def _contact_support(distance: float | None, thresholds: ContactThresholds) -> str:
    if distance is None:
        return "unmeasured"
    if distance <= thresholds.enter_distance:
        return "contact"
    if distance > thresholds.exit_distance:
        return "no_contact"
    return "indeterminate"


def _measurement(
    comparison: ConformerComparison,
    name_a: str,
    name_b: str,
    region_definitions: dict[str, tuple[Sequence[int], Sequence[int] | None]] | None,
) -> dict[str, Any]:
    regions = comparison.alignment.regions
    if name_a not in regions or name_b not in regions:
        raise KeyError(f"Regions {name_a!r} and {name_b!r} must both be measured")
    region_a = regions[name_a]
    region_b = regions[name_b]
    atom_pairs = {
        name_a: [[int(ref), int(mob)] for ref, mob in zip(region_a.reference_keys, region_a.mobile_keys)],
        name_b: [[int(ref), int(mob)] for ref, mob in zip(region_b.reference_keys, region_b.mobile_keys)],
    }
    if region_definitions is not None and name_a in region_definitions and name_b in region_definitions:
        expected = {
            name: {int(key) for key in region_definitions[name][0]}
            for name in (name_a, name_b)
        }
    else:
        expected = {name_a: set(), name_b: set()}
    matched = {
        name_a: set(int(key) for key in region_a.reference_keys),
        name_b: set(int(key) for key in region_b.reference_keys),
    }
    missing_reference_keys = {
        name: sorted(expected[name] - matched[name]) for name in (name_a, name_b)
    }

    structure = comparison.conformer.structure
    coords_a = [structure.atom(int(key)).xyz for key in region_a.mobile_keys]
    coords_b = [structure.atom(int(key)).xyz for key in region_b.mobile_keys]
    distance_value: float | None = None
    closest_pair: list[int] | None = None
    if coords_a and coords_b:
        matrix_a = np.vstack(coords_a)
        matrix_b = np.vstack(coords_b)
        deltas = matrix_a[:, None, :] - matrix_b[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        flat_index = int(np.argmin(distances))
        row, col = np.unravel_index(flat_index, distances.shape)
        distance_value = float(distances[row, col])
        closest_pair = [int(region_a.reference_keys[row]), int(region_b.reference_keys[col])]

    return {
        "distance": distance_value,
        "closest_pair": closest_pair,
        "atom_pairs": atom_pairs,
        "missing_reference_keys": missing_reference_keys,
        "candidate_id": comparison.candidate_id,
    }


def _event(point: ContactPoint, event_type: str, *, reason: str | None = None) -> dict[str, Any]:
    event = {
        "index": point.index,
        "conformer_name": point.conformer_name,
        "source_path": point.source_path,
        "event": event_type,
        "state": point.state,
        "distance": point.distance,
        "closest_pair": point.closest_pair,
        "candidate_id": point.candidate_id,
        "decision_id": point.decision_id,
    }
    if reason is not None:
        event["reason"] = reason
    return event


def _close_episode(
    trace: ContactTrace,
    indices: Sequence[int],
    points: Sequence[ContactPoint],
    *,
    end_reason: str,
) -> None:
    episode_points = [points[index] for index in indices]
    confirmed = sum(1 for point in episode_points if point.decision_id is not None)
    trace.episodes.append(
        ContactEpisode(
            episode_id=f"episode-{len(trace.episodes) + 1:03d}",
            start_index=indices[0],
            end_index=indices[-1],
            start_conformer=episode_points[0].conformer_name,
            end_conformer=episode_points[-1].conformer_name,
            observed_frame_count=len(episode_points) - confirmed,
            confirmed_frame_count=confirmed,
            point_indices=list(indices),
            end_reason=end_reason,
        )
    )
