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
    evidence_segments: list[dict[str, Any]] = field(default_factory=list)
    events: list[ContactEvent] = field(default_factory=list)
    status: str = "active"
    current_observed_run: int = 0

    def to_dict(self) -> dict[str, Any]:
        gaps = [
            segment
            for segment in self.evidence_segments
            if segment["evidence_state"] in ("missing_atoms", "interpretation_disputed")
        ]
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
            "unresolved_gap_duration": sum(segment["end_index"] - segment["start_index"] + 1 for segment in gaps),
            "has_unresolved_gap": bool(gaps),
            "continuous_observed_contact": not gaps,
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
    segment: dict[str, Any] | None = None

    def close_segment(end_index: int) -> None:
        nonlocal segment
        if segment is not None:
            segment["end_index"] = end_index
            if active is not None:
                active.evidence_segments.append(segment)
            segment = None

    def append_segment(start_index: int, state: str) -> None:
        nonlocal segment
        close_segment(start_index - 1)
        segment = {
            "start_index": start_index,
            "end_index": start_index,
            "evidence_state": state,
        }

    for contact_point in raw_points:
        index = contact_point.index
        evidence = contact_point.evidence_state
        if evidence in ("observed_contact", "confirmed_contact"):
            if active is None:
                active = ContactInterval(start_index=index)
                announced = False
            append_segment(index, evidence)
            active.observed_contact_indices.append(index)
            active.current_observed_run += 1
            if evidence == "confirmed_contact":
                active.confirmed_contact_indices.append(index)
            if not announced and active.current_observed_run >= rule.min_contact_duration:
                event = ContactEvent(
                    "formed",
                    index,
                    active.start_index,
                    "Contact entered the formation distance and satisfied the minimum duration.",
                )
                active.events.append(event)
                events.append(event)
                announced = True
        elif evidence in ("missing_atoms", "interpretation_disputed"):
            if active is not None:
                append_segment(index, evidence)
                if not announced:
                    close_segment(index - 1)
                    active = None
                else:
                    active.current_observed_run = 0
                    active.status = "unresolved"
            contact_point.state = "unresolved"
        elif evidence == "hysteresis_hold":
            if active is not None:
                append_segment(index, "hysteresis_hold")
                active.hysteresis_hold_indices.append(index)
                active.status = "active"
            else:
                contact_point.state = "no_contact"
        elif evidence == "no_contact" and active is not None:
            close_segment(index - 1)
            if len(active.observed_contact_indices) >= rule.min_contact_duration:
                active.end_index = index
                active.status = "lost"
                event = ContactEvent(
                    "lost",
                    index,
                    active.start_index,
                    "Contact reached the release distance; unresolved frames did not close it.",
                )
                active.events.append(event)
                events.append(event)
                intervals.append(active)
            active = None
            announced = False
            segment = None

    close_segment(len(raw_points) - 1)
    if active is not None:
        if len(active.observed_contact_indices) >= rule.min_contact_duration:
            active.status = "active"
            intervals.append(active)
    return ContactAnalysis(rule=rule, points=raw_points, intervals=intervals, events=events)


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
    if distance is None:
        evidence = "missing_atoms"
        state = "unresolved"
        explanation = "Required atoms are absent; contact can neither be formed nor released at this frame."
    elif low is not None and high is not None and low <= rule.enter_distance and high >= rule.exit_distance:
        evidence = "interpretation_disputed"
        state = "unresolved"
        explanation = "Alternative atom correspondences support both contact and no-contact interpretations."
    elif low is not None and high is not None and high < rule.exit_distance:
        evidence = "confirmed_contact" if manually_confirmed else "observed_contact"
        state = "contact"
        explanation = "Every available correspondence places the regions within the release distance."
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
        candidate_id=point.candidate_id,
        correspondence_status=point.correspondence_status,
        decision_id=point.decision_id,
        source_path=point.source_path,
        conformer_name=point.conformer_name,
        explanation=explanation,
        alternative_measurements=getattr(point, "alternative_measurements", []),
    )
