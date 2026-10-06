from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .contacts import ContactAnalysis, ContactRule, analyze_contacts
from .relative_motion import RelativeMotionSeries

_CONTACT_EVIDENCE = ("observed_contact", "confirmed_contact", "hysteresis_hold")
_UNKNOWN_EVIDENCE = ("missing_atoms", "interpretation_disputed", "not_observed")


def _pair_key(region_a: str, region_b: str) -> tuple[str, str]:
    return tuple(sorted((region_a, region_b)))


def _member_label(rule: ContactRule) -> str:
    return rule.region_a + "__" + rule.region_b


@dataclass(eq=False)
class JointContactRule:
    members: list[ContactRule]
    max_handoff_gap: int = 0
    name: str | None = None

    def __post_init__(self) -> None:
        self.members = [ContactRule.from_value(member) for member in self.members]
        if len(self.members) < 2:
            raise ValueError("A joint contact rule needs at least two member pairs")
        labels = [_pair_key(member.region_a, member.region_b) for member in self.members]
        if len(set(labels)) != len(labels):
            raise ValueError("Joint contact members must be distinct region pairs")
        if self.max_handoff_gap < 0:
            raise ValueError("max_handoff_gap must be non-negative")
        if self.name is None:
            self.name = "+".join(sorted(_member_label(member) for member in self.members))

    @classmethod
    def from_value(cls, value):
        if isinstance(value, JointContactRule):
            return value
        return cls(
            members=[ContactRule.from_value(member) for member in value["members"]],
            max_handoff_gap=int(value.get("max_handoff_gap", 0)),
            name=value.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "members": [member.to_dict() for member in self.members],
            "max_handoff_gap": self.max_handoff_gap,
        }


@dataclass(eq=False)
class JointContactPoint:
    index: int
    state: str
    member_states: dict[str, str]
    blocking_members: list[dict[str, Any]] = field(default_factory=list)
    unknown_members: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "state": self.state,
            "member_states": dict(self.member_states),
            "blocking_members": self.blocking_members,
            "unknown_members": self.unknown_members,
        }


@dataclass(eq=False)
class JointWindow:
    start_index: int
    end_index: int
    status: str
    proven_contact_indices: list[int] = field(default_factory=list)
    handoff_indices: list[int] = field(default_factory=list)
    handoff_details: list[dict[str, Any]] = field(default_factory=list)
    member_support: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "status": self.status,
            "proven_contact_indices": list(self.proven_contact_indices),
            "proven_duration": len(self.proven_contact_indices),
            "handoff_indices": list(self.handoff_indices),
            "handoff_duration": len(self.handoff_indices),
            "handoff_details": self.handoff_details,
            "member_support": self.member_support,
        }


@dataclass(eq=False)
class JointContactAnalysis:
    rule: JointContactRule
    points: list[JointContactPoint] = field(default_factory=list)
    windows: list[JointWindow] = field(default_factory=list)
    undetermined_ranges: list[dict[str, Any]] = field(default_factory=list)
    blocking_segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "molecular-toolkit-joint-contact-analysis",
            "version": 1,
            "rule": self.rule.to_dict(),
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
            "windows": [window.to_dict() for window in self.windows],
            "proven_window_count": len(self.windows),
            "proven_duration": sum(len(window.proven_contact_indices) for window in self.windows),
            "undetermined_ranges": self.undetermined_ranges,
            "undetermined_duration": sum(
                item["end_index"] - item["start_index"] + 1 for item in self.undetermined_ranges
            ),
            "blocking_segments": self.blocking_segments,
            "blocking_duration": sum(
                item["end_index"] - item["start_index"] + 1 for item in self.blocking_segments
            ),
        }


def analyze_joint_contacts(
    analyses: Sequence[ContactAnalysis],
    rule: JointContactRule | dict[str, Any],
) -> JointContactAnalysis:
    """Combine per-pair contact analyses into joint-action time coverage.

    Joint contact at a frame requires every member pair to hold contact at that
    same frame; observations from different pairs are never stitched across
    frames. Frames where a member is unknown (missing atoms or disputed
    interpretation) only make the dependent joint conclusion undetermined.
    """
    rule = JointContactRule.from_value(rule)
    member_analyses = [_find_member_analysis(analyses, member) for member in rule.members]
    labels = [_member_label(member) for member in rule.members]
    points = _joint_points(rule, member_analyses, labels)
    result = JointContactAnalysis(rule=rule, points=points)
    _build_windows(result, rule, member_analyses, labels)
    return result


def analyze_joint_contact_scenarios(
    series: Sequence[RelativeMotionSeries],
    scenarios: Sequence[JointContactRule | dict[str, Any]],
) -> dict[str, Any]:
    """Run several threshold/handoff scenarios on the same relative-motion series."""
    results = []
    for scenario in scenarios:
        joint_rule = JointContactRule.from_value(scenario)
        analyses = [
            analyze_contacts(_find_series(series, member), member)
            for member in joint_rule.members
        ]
        joint = analyze_joint_contacts(analyses, joint_rule)
        results.append(joint.to_dict())
    return {
        "format": "molecular-toolkit-joint-contact-scenarios",
        "version": 1,
        "scenarios": results,
        "summary": [
            {
                "name": payload["rule"]["name"],
                "max_handoff_gap": payload["rule"]["max_handoff_gap"],
                "members": payload["rule"]["members"],
                "proven_window_count": payload["proven_window_count"],
                "proven_duration": payload["proven_duration"],
                "undetermined_duration": payload["undetermined_duration"],
                "blocking_duration": payload["blocking_duration"],
            }
            for payload in results
        ],
    }


def save_joint_contact_analysis(analysis: JointContactAnalysis, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find_member_analysis(
    analyses: Sequence[ContactAnalysis], member: ContactRule
) -> ContactAnalysis:
    key = _pair_key(member.region_a, member.region_b)
    matches = [
        analysis
        for analysis in analyses
        if _pair_key(analysis.rule.region_a, analysis.rule.region_b) == key
        and analysis.rule.enter_distance == member.enter_distance
        and analysis.rule.exit_distance == member.exit_distance
        and analysis.rule.min_contact_duration == member.min_contact_duration
    ]
    if not matches:
        raise KeyError(
            "No contact analysis for joint member "
            + _member_label(member)
            + " with enter=%g exit=%g min_duration=%d"
            % (member.enter_distance, member.exit_distance, member.min_contact_duration)
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple contact analyses match joint member "
            + _member_label(member)
            + "; pass exactly one analysis per member rule"
        )
    return matches[0]


def _find_series(series: Sequence[RelativeMotionSeries], member: ContactRule) -> RelativeMotionSeries:
    key = _pair_key(member.region_a, member.region_b)
    for item in series:
        if _pair_key(item.region_a, item.region_b) == key:
            return item
    raise KeyError("No relative motion series for region pair " + _member_label(member))


def _joint_points(
    rule: JointContactRule,
    member_analyses: list[ContactAnalysis],
    labels: list[str],
) -> list[JointContactPoint]:
    points_by_member = [
        {point.index: point for point in analysis.points} for analysis in member_analyses
    ]
    indices = sorted({index for points in points_by_member for index in points})
    joint_points: list[JointContactPoint] = []
    for index in indices:
        member_states: dict[str, str] = {}
        blocking: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        for label, member, points in zip(labels, rule.members, points_by_member):
            point = points.get(index)
            if point is None:
                member_states[label] = "not_observed"
                unknown.append(
                    {
                        "member": label,
                        "evidence_state": "not_observed",
                        "reason": "This pair has no observation at the frame; the joint "
                        "conclusion depending on it stays unknown.",
                    }
                )
                continue
            member_states[label] = point.evidence_state
            if point.evidence_state == "no_contact":
                blocking.append(
                    {
                        "member": label,
                        "evidence_state": point.evidence_state,
                        "distance": point.distance,
                        "exit_distance": member.exit_distance,
                        "reason": point.explanation,
                    }
                )
            elif point.evidence_state in _UNKNOWN_EVIDENCE:
                unknown.append(
                    {
                        "member": label,
                        "evidence_state": point.evidence_state,
                        "missing_atoms": {
                            name: list(keys) for name, keys in point.missing_atoms.items()
                        },
                        "reason": point.explanation,
                    }
                )
        if blocking:
            state = "joint_separated"
        elif unknown:
            state = "joint_unknown"
        else:
            state = "joint_contact"
        joint_points.append(
            JointContactPoint(
                index=index,
                state=state,
                member_states=member_states,
                blocking_members=blocking,
                unknown_members=unknown,
            )
        )
    return joint_points


def _build_windows(
    result: JointContactAnalysis,
    rule: JointContactRule,
    member_analyses: list[ContactAnalysis],
    labels: list[str],
) -> None:
    points = result.points
    position = 0
    while position < len(points):
        point = points[position]
        if point.state == "joint_contact":
            position = _consume_window(result, rule, member_analyses, labels, position)
        elif point.state == "joint_unknown":
            position = _consume_undetermined(result, position)
        else:
            position = _consume_blocking(result, position)


def _consume_window(
    result: JointContactAnalysis,
    rule: JointContactRule,
    member_analyses: list[ContactAnalysis],
    labels: list[str],
    start: int,
) -> int:
    points = result.points
    count = len(points)
    proven = [points[start].index]
    handoff: list[int] = []
    handoff_details: list[dict[str, Any]] = []
    cursor = start + 1
    status = "active"
    while cursor < count:
        point = points[cursor]
        if point.state == "joint_contact":
            proven.append(point.index)
            cursor += 1
            continue
        if point.state == "joint_unknown":
            status = "interrupted_unknown"
            break
        lookahead = cursor
        while lookahead < count and points[lookahead].state == "joint_separated":
            lookahead += 1
        run = lookahead - cursor
        if (
            run <= rule.max_handoff_gap
            and lookahead < count
            and points[lookahead].state == "joint_contact"
        ):
            for offset in range(cursor, lookahead):
                handoff.append(points[offset].index)
                handoff_details.append(
                    {
                        "index": points[offset].index,
                        "blocking_members": points[offset].blocking_members,
                        "reason": "Brief separation within the allowed handoff gap; the joint "
                        "window continues but this frame is not proven joint contact.",
                    }
                )
            cursor = lookahead
            continue
        status = "closed"
        break
    window = JointWindow(
        start_index=proven[0],
        end_index=proven[-1],
        status=status,
        proven_contact_indices=proven,
        handoff_indices=handoff,
        handoff_details=handoff_details,
        member_support=_member_support(member_analyses, labels, proven[0], proven[-1]),
    )
    result.windows.append(window)
    return cursor


def _consume_undetermined(result: JointContactAnalysis, start: int) -> int:
    points = result.points
    count = len(points)
    cursor = start
    frames = []
    while cursor < count and points[cursor].state == "joint_unknown":
        point = points[cursor]
        frames.append(
            {
                "index": point.index,
                "unknown_members": point.unknown_members,
                "member_states": dict(point.member_states),
            }
        )
        cursor += 1
    result.undetermined_ranges.append(
        {
            "start_index": points[start].index,
            "end_index": points[cursor - 1].index,
            "duration": cursor - start,
            "frames": frames,
            "reason": "At least one member pair is unknown at these frames (missing atoms or "
            "disputed interpretation); joint action can neither be confirmed nor ruled out, "
            "and neighbouring proven windows are not merged across this range.",
        }
    )
    return cursor


def _consume_blocking(result: JointContactAnalysis, start: int) -> int:
    points = result.points
    count = len(points)
    cursor = start
    frames = []
    while cursor < count and points[cursor].state == "joint_separated":
        point = points[cursor]
        frames.append(
            {
                "index": point.index,
                "blocking_members": point.blocking_members,
                "member_states": dict(point.member_states),
            }
        )
        cursor += 1
    result.blocking_segments.append(
        {
            "start_index": points[start].index,
            "end_index": points[cursor - 1].index,
            "duration": cursor - start,
            "frames": frames,
            "reason": "At least one member pair is confirmed separated beyond its release "
            "distance; no joint window can span these frames.",
        }
    )
    return cursor


def _member_support(
    member_analyses: list[ContactAnalysis],
    labels: list[str],
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    support: dict[str, Any] = {}
    for label, analysis in zip(labels, member_analyses):
        overlapping = []
        for interval in analysis.intervals:
            interval_end = interval.end_index if interval.end_index is not None else end_index
            if interval.start_index > end_index or interval_end < start_index:
                continue
            payload = interval.to_dict()
            payload["overlap_start_index"] = max(interval.start_index, start_index)
            payload["overlap_end_index"] = min(interval_end, end_index)
            overlapping.append(payload)
        frame_states = {
            point.index: point.evidence_state
            for point in analysis.points
            if start_index <= point.index <= end_index
        }
        support[label] = {
            "intervals": overlapping,
            "frame_states": frame_states,
        }
    return support
