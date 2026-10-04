from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .relative_motion import RelativeMotionSeries


@dataclass(eq=False)
class ContactRule:
    region_a: str
    region_b: str
    enter_distance: float
    exit_distance: float | None = None
    min_contact_duration: int = 1

    def __post_init__(self) -> None:
        if self.enter_distance < 0:
            raise ValueError("enter_distance must be non-negative")
        if self.exit_distance is None:
            self.exit_distance = self.enter_distance
        if self.exit_distance < self.enter_distance:
            raise ValueError("exit_distance must be greater than or equal to enter_distance")
        if self.min_contact_duration < 1:
            raise ValueError("min_contact_duration must be at least 1")

    @classmethod
    def from_value(cls, value: "ContactRule | dict[str, Any]") -> "ContactRule":
        if isinstance(value, ContactRule):
            return value
        return cls(
            region_a=value["region_a"],
            region_b=value["region_b"],
            enter_distance=float(value["enter_distance"]),
            exit_distance=None if value.get("exit_distance") is None else float(value["exit_distance"]),
            min_contact_duration=int(value.get("min_contact_duration", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_a": self.region_a,
            "region_b": self.region_b,
            "enter_distance": self.enter_distance,
            "exit_distance": self.exit_distance,
            "min_contact_duration": self.min_contact_duration,
        }


@dataclass(eq=False)
class ContactPoint:
    index: int
    state: str
    evidence_state: str
    distance: float | None
    distance_range: tuple[float | None, float | None] | None
    atom_pairs: dict[str, list[list[int]]]
    atom_coverage: dict[str, dict[str, int | None]] | None
    candidate_id: str | None
    correspondence_status: str
    decision_id: str | None
    source_path: str | None
    conformer_name: str
    explanation: str
    alternative_measurements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "state": self.state,
            "evidence_state": self.evidence_state,
            "distance": self.distance,
            "distance_range": list(self.distance_range) if self.distance_range is not None else None,
            "atom_pairs": self.atom_pairs,
            "atom_coverage": self.atom_coverage,
            "candidate_id": self.candidate_id,
            "correspondence_status": self.correspondence_status,
            "decision_id": self.decision_id,
            "source_path": self.source_path,
            "conformer_name": self.conformer_name,
            "explanation": self.explanation,
            "alternative_measurements": self.alternative_measurements,
        }


@dataclass(eq=False)
class ContactEvent:
    event_type: str
    index: int
    contact_start_index: int | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "index": self.index,
            "contact_start_index": self.contact_start_index,
            "message": self.message,
        }


@dataclass(eq=False)
class ContactInterval:
    start_index: int
    end_index: int | None = None
    observed_contact_indices: list[int] = field(default_factory=list)
    confirmed_contact_indices: list[int] = field(default_factory=list)
    hysteresis_hold_indices: list[int] = field(default_factory=list)
    unresolved_indices: list[int] = field(default_factory=list)
    evidence_segments: list[dict[str, Any]] = field(default_factory=list)
    events: list[ContactEvent] = field(default_factory=list)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        gap_states = (
            "missing_all_atoms",
            "missing_partial_atoms",
            "missing_atoms",
            "interpretation_disputed",
        )
        gaps = [segment for segment in self.evidence_segments if segment["evidence_state"] in gap_states]
        notes: list[str] = []
        if gaps:
            kinds = sorted({segment["evidence_state"] for segment in gaps})
            notes.append(
                "Unresolved frames (" + ", ".join(kinds) + ") are retained as unknown evidence: they neither "
                "count as zero-distance contact nor as confirmed separation, and they do not release the contact."
            )
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "status": self.status,
            "observed_contact_duration": len(self.observed_contact_indices),
            "confirmed_contact_duration": len(self.confirmed_contact_indices),
            "hysteresis_hold_duration": len(self.hysteresis_hold_indices),
            "sustained_duration": len(self.observed_contact_indices) + len(self.hysteresis_hold_indices),
            "observed_contact_indices": list(self.observed_contact_indices),
            "confirmed_contact_indices": list(self.confirmed_contact_indices),
            "hysteresis_hold_indices": list(self.hysteresis_hold_indices),
            "unresolved_indices": list(self.unresolved_indices),
            "unresolved_gap_duration": sum(segment["end_index"] - segment["start_index"] + 1 for segment in gaps),
            "has_unresolved_gap": bool(gaps),
            "continuous_observed_contact": not gaps,
            "notes": notes,
            "evidence_segments": self.evidence_segments,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(eq=False)
class ContactAnalysis:
    rule: ContactRule
    points: list[ContactPoint] = field(default_factory=list)
    intervals: list[ContactInterval] = field(default_factory=list)
    events: list[ContactEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "molecular-toolkit-contact-analysis",
            "version": 1,
            "rule": self.rule.to_dict(),
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
            "events": [event.to_dict() for event in self.events],
            "intervals": [interval.to_dict() for interval in self.intervals],
        }


def analyze_contacts(series: RelativeMotionSeries, rule: ContactRule | dict[str, Any]) -> ContactAnalysis:
    rule = ContactRule.from_value(rule)
    _assert_series_regions(series, rule)
    raw_points = [_classify_point(point, rule) for point in series.points]
    intervals: list[ContactInterval] = []
    events: list[ContactEvent] = []
    active: ContactInterval | None = None
    announced = False
    segments: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None

    def append_segment(index: int, state: str) -> None:
        nonlocal current_segment
        if current_segment is not None and current_segment["evidence_state"] == state:
            current_segment["end_index"] = index
            return
        if current_segment is not None:
            segments.append(current_segment)
        current_segment = {
            "start_index": index,
            "end_index": index,
            "evidence_state": state,
        }

    contact_evidence = ("observed_contact", "confirmed_contact")
    unresolved_evidence = (
        "missing_all_atoms",
        "missing_partial_atoms",
        "missing_atoms",
        "interpretation_disputed",
    )

    for contact_point in raw_points:
        index = contact_point.index
        evidence = contact_point.evidence_state
        if evidence in contact_evidence:
            append_segment(index, evidence)
            if active is None:
                active = ContactInterval(start_index=index)
                announced = False
            active.observed_contact_indices.append(index)
            if evidence == "confirmed_contact":
                active.confirmed_contact_indices.append(index)
            if not announced and _trailing_observed_run(raw_points, index) >= rule.min_contact_duration:
                event = ContactEvent(
                    "formed",
                    index,
                    active.start_index,
                    "Contact entered the formation distance and satisfied the minimum duration.",
                )
                active.events.append(event)
                events.append(event)
                announced = True
            active.status = "active"
        elif evidence in unresolved_evidence:
            append_segment(index, evidence)
            if active is not None:
                active.unresolved_indices.append(index)
                if not announced:
                    active = None
                    # Unknown frames hold the outcome open; an already
                    # announced contact is neither lost nor bridged.
                if active is not None:
                    active.status = "unresolved"
            contact_point.state = "unresolved"
        elif evidence == "hysteresis_hold":
            append_segment(index, "hysteresis_hold")
            if active is not None:
                active.hysteresis_hold_indices.append(index)
                if announced:
                    active.status = "active"
            else:
                contact_point.state = "no_contact"
        elif evidence == "no_contact":
            append_segment(index, "no_contact")
            if active is not None:
                if announced:
                    active.end_index = index
                    active.status = "lost"
                    message = "Adopted correspondence placed the regions at or beyond the release distance."
                    event = ContactEvent("lost", index, active.start_index, message)
                    active.events.append(event)
                    events.append(event)
                    intervals.append(active)
                active = None
                announced = False

    if current_segment is not None:
        segments.append(current_segment)
    if active is not None:
        if announced:
            active.status = "unresolved" if _ends_unresolved(raw_points) and active.unresolved_indices else "active"
            intervals.append(active)
    _attach_segments(intervals, segments, len(raw_points) - 1)
    return ContactAnalysis(rule=rule, points=raw_points, intervals=intervals, events=events)


def _trailing_observed_run(points, index):
    """Count consecutive directly observed/held frames ending at index.

    Contact frames and hysteresis-hold frames both contribute to the
    minimum formation duration; unresolved frames interrupt the run.
    """
    run = 0
    cursor = index
    while cursor >= 0:
        if points[cursor].evidence_state in (
            "observed_contact",
            "confirmed_contact",
            "hysteresis_hold",
        ):
            run += 1
        else:
            break
        cursor -= 1
    return run


def _ends_unresolved(points):
    return points[-1].evidence_state in (
        "missing_all_atoms",
        "missing_partial_atoms",
        "missing_atoms",
        "interpretation_disputed",
    )


def _attach_segments(intervals, segments, last_index):
    for interval in intervals:
        end = interval.end_index if interval.end_index is not None else last_index
        interval.evidence_segments = [
            segment
            for segment in segments
            if interval.start_index <= segment["start_index"] and segment["end_index"] <= end
        ]


def contact_analyses_for_series(
    series: RelativeMotionSeries,
    rules: Sequence[ContactRule | dict[str, Any]],
) -> list[ContactAnalysis]:
    return [analyze_contacts(series, rule) for rule in rules if _matches_series(series, rule)]


def save_contact_analysis(analysis: ContactAnalysis, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contact_decision_preview(before: ContactAnalysis, after: ContactAnalysis) -> dict[str, Any]:
    before_events = [event.to_dict() for event in before.events]
    after_events = [event.to_dict() for event in after.events]
    before_intervals = [interval.to_dict() for interval in before.intervals]
    after_intervals = [interval.to_dict() for interval in after.intervals]
    return {
        "rule": after.rule.to_dict(),
        "before": {"events": before_events, "intervals": before_intervals},
        "after": {"events": after_events, "intervals": after_intervals},
        "events_changed": before_events != after_events,
        "durations_changed": [
            index
            for index, (old, new) in enumerate(zip(before_intervals, after_intervals))
            if (old["observed_contact_duration"], old["confirmed_contact_duration"], old["unresolved_gap_duration"])
            != (new["observed_contact_duration"], new["confirmed_contact_duration"], new["unresolved_gap_duration"])
        ],
        "interval_count_changed": len(before_intervals) != len(after_intervals),
    }


def _assert_series_regions(series: RelativeMotionSeries, rule: ContactRule | dict[str, Any]) -> None:
    rule = ContactRule.from_value(rule)
    if not _matches_series(series, rule):
        raise KeyError(f"Contact rule {rule.region_a!r}/{rule.region_b!r} does not match series")


def _matches_series(series: RelativeMotionSeries, rule: ContactRule | dict[str, Any]) -> bool:
    parsed = ContactRule.from_value(rule)
    return (series.region_a, series.region_b) in [(parsed.region_a, parsed.region_b), (parsed.region_b, parsed.region_a)]


def _classify_point(point, rule: ContactRule) -> ContactPoint:
    distance = point.distance
    low, high = (None, None) if point.distance_range is None else point.distance_range
    manually_confirmed = point.decision_id is not None
    expected_a = getattr(point, "region_a_expected_atoms", None)
    observed_a = getattr(point, "region_a_observed_atoms", None)
    expected_b = getattr(point, "region_b_expected_atoms", None)
    observed_b = getattr(point, "region_b_observed_atoms", None)
    coverage = _coverage_dict(point, expected_a, observed_a, expected_b, observed_b)
    missing = _missing_regions(expected_a, observed_a, expected_b, observed_b)
    if missing == "all":
        evidence = "missing_all_atoms"
        state = "unresolved"
        explanation = (
            "All atoms of at least one contact region are absent at this frame; "
            "contact status is unknown and must not be read as zero-distance contact or as separation."
        )
    elif missing == "partial":
        evidence = "missing_partial_atoms"
        state = "unresolved"
        explanation = (
            "Only part of a contact region is observed at this frame; the measured distance is not a complete "
            "basis, so contact status stays unknown instead of being forced to contact or separation."
        )
    elif distance is None:
        evidence = "missing_atoms"
        state = "unresolved"
        explanation = "Required atoms are absent; contact can neither be formed nor released at this frame."
    elif low is not None and high is not None and low <= rule.enter_distance and high >= rule.exit_distance:
        evidence = "interpretation_disputed"
        state = "unresolved"
        explanation = "Alternative atom correspondences support both contact and no-contact interpretations."
    elif low is not None and high is not None and low < rule.enter_distance <= high:
        evidence = "interpretation_disputed"
        state = "unresolved"
        explanation = (
            "Alternative atom correspondences place the regions on different sides of the formation "
            "threshold; contact status is unknown."
        )
    elif low is not None and high is not None and low < rule.exit_distance <= high:
        evidence = "interpretation_disputed"
        state = "unresolved"
        explanation = (
            "Alternative atom correspondences place the regions on different sides of the release "
            "threshold; separation cannot be asserted."
        )
    elif distance <= rule.enter_distance:
        evidence = "confirmed_contact" if manually_confirmed else "observed_contact"
        state = "contact"
        explanation = "Adopted atom correspondence places the regions at or within the formation distance."
    elif distance >= rule.exit_distance:
        evidence = "no_contact"
        state = "no_contact"
        explanation = "Adopted atom correspondence places the regions at or beyond the release distance."
    else:
        evidence = "hysteresis_hold"
        state = "contact_held"
        explanation = "Distance is between formation and release thresholds; an existing contact is retained."
    return ContactPoint(
        index=point.index,
        state=state,
        evidence_state=evidence,
        distance=distance,
        distance_range=point.distance_range,
        atom_pairs=point.atom_pairs,
        atom_coverage=coverage,
        candidate_id=point.candidate_id,
        correspondence_status=point.correspondence_status,
        decision_id=point.decision_id,
        source_path=point.source_path,
        conformer_name=point.conformer_name,
        explanation=explanation,
        alternative_measurements=getattr(point, "alternative_measurements", []),
    )


def _missing_regions(expected_a, observed_a, expected_b, observed_b) -> str | None:
    """Return 'all' when a region vanished, 'partial' for incomplete coverage."""
    states = [_coverage_state(expected_a, observed_a), _coverage_state(expected_b, observed_b)]
    if "all" in states:
        return "all"
    if "partial" in states:
        return "partial"
    return None


def _coverage_state(expected, observed) -> str | None:
    if expected is None:
        return None
    if observed is None or observed <= 0:
        return "all"
    if observed < expected:
        return "partial"
    return None


def _coverage_dict(point, expected_a, observed_a, expected_b, observed_b):
    if expected_a is None and expected_b is None:
        return getattr(point, "atom_coverage", None)

    def entry(expected, observed):
        return {
            "expected_atoms": expected,
            "observed_atoms": observed,
            "missing_atoms": None if expected is None else expected - (observed or 0),
        }

    return {
        point.region_a: entry(expected_a, observed_a),
        point.region_b: entry(expected_b, observed_b),
    }
