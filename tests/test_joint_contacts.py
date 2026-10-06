import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    AtomCorrespondence,
    ComparisonProject,
    Conformer,
    ConformerCorrespondenceSpec,
    ContactRule,
    JointContactRule,
    Structure,
    analyze_contacts,
    analyze_joint_contact_scenarios,
    analyze_joint_contacts,
    analyze_relative_motion,
    compare_ensemble,
    load_comparison_project,
    run_comparison_project,
    save_comparison_project,
    save_structure,
    transform_structure,
)

ENTER_DISTANCE = 0.3
EXIT_DISTANCE = 0.6
CORE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
OFFSETS = [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
LEFT_BASE = (3.0, 0.0, 0.0)
RIGHT_BASE = (4.0, 0.0, 0.0)
TOP_BASE = (3.0, 5.0, 0.0)
LEFT_KEYS = [4, 5, 6]
RIGHT_KEYS = [7, 8, 9]
TOP_KEYS = [10, 11, 12]
ALL_KEYS = list(range(13))

# frame: (left-right distance, left-top distance, right keys present, top keys present)
FRAMES = [
    (0.2, 0.9, RIGHT_KEYS, TOP_KEYS),
    (0.2, 0.2, RIGHT_KEYS, TOP_KEYS),
    (0.9, 0.2, RIGHT_KEYS, TOP_KEYS),
    (0.2, 0.2, RIGHT_KEYS, TOP_KEYS),
    (0.2, None, RIGHT_KEYS, [10, 11]),
    (0.2, 0.2, RIGHT_KEYS, TOP_KEYS),
    (0.2, 0.2, RIGHT_KEYS, TOP_KEYS),
    (0.2, 0.9, RIGHT_KEYS, TOP_KEYS),
]

EXPECTED_JOINT_STATES = [
    "joint_separated",
    "joint_contact",
    "joint_separated",
    "joint_contact",
    "joint_unknown",
    "joint_contact",
    "joint_contact",
    "joint_separated",
]


def _xyz(base, offsets):
    return [tuple(np.asarray(base) + np.asarray(offset)) for offset in offsets]


def make_reference():
    coordinates = CORE + _xyz(LEFT_BASE, OFFSETS) + _xyz(RIGHT_BASE, OFFSETS) + _xyz(TOP_BASE, OFFSETS)
    return Structure([Atom(key, "C", coordinates[key], serial=key + 1) for key in ALL_KEYS])


def make_mobile(lr_distance, lt_distance, right_keys, top_keys):
    coordinates = list(CORE) + _xyz(LEFT_BASE, OFFSETS)
    keys = list(range(7))
    right_base = np.asarray(LEFT_BASE) + np.asarray((lr_distance, 0.0, 0.0))
    right_xyz = _xyz(tuple(right_base), OFFSETS)
    for key in right_keys:
        coordinates.append(right_xyz[RIGHT_KEYS.index(key)])
        keys.append(key)
    top_base = np.asarray(LEFT_BASE) + np.asarray((0.0, lt_distance if lt_distance is not None else 0.2, 0.0))
    top_xyz = _xyz(tuple(top_base), OFFSETS)
    for key in top_keys:
        coordinates.append(top_xyz[TOP_KEYS.index(key)])
        keys.append(key)
    atoms = [Atom(index, "C", coordinates[index], serial=keys[index] + 1) for index in range(len(keys))]
    return Structure(atoms), keys


def make_correspondence(keys):
    present = set(keys)
    return AtomCorrespondence(
        pairs=[(key, index) for index, key in enumerate(keys)],
        unmatched_reference=[key for key in ALL_KEYS if key not in present],
        unmatched_mobile=[],
        ambiguous_identities={},
    )


def make_conformers(transform=None):
    conformers = []
    correspondences = []
    for index, (lr_distance, lt_distance, right_keys, top_keys) in enumerate(FRAMES):
        mobile, keys = make_mobile(lr_distance, lt_distance, right_keys, top_keys)
        if transform is not None:
            rotation, translation = transform
            mobile = transform_structure(mobile, rotation, translation)
        conformers.append(Conformer(name="frame-%d" % index, structure=mobile))
        correspondences.append(make_correspondence(keys))
    return conformers, correspondences


def make_series(transform=None):
    reference = make_reference()
    conformers, correspondences = make_conformers(transform)
    comparisons = compare_ensemble(
        reference,
        list(zip(conformers, correspondences)),
        [0, 1, 2, 3],
        {"left": (LEFT_KEYS, None), "right": (RIGHT_KEYS, None), "top": (TOP_KEYS, None)},
    )
    return [
        analyze_relative_motion(comparisons, "left", "right"),
        analyze_relative_motion(comparisons, "left", "top"),
    ]


def make_rule(max_handoff_gap=0, enter=ENTER_DISTANCE):
    return JointContactRule(
        members=[
            ContactRule("left", "right", enter, EXIT_DISTANCE),
            ContactRule("left", "top", enter, EXIT_DISTANCE),
        ],
        max_handoff_gap=max_handoff_gap,
    )


def run_joint_analysis(transform=None, max_handoff_gap=0, enter=ENTER_DISTANCE):
    rule = make_rule(max_handoff_gap=max_handoff_gap, enter=enter)
    analyses = [analyze_contacts(series, member) for series, member in zip(make_series(transform), rule.members)]
    return analyze_joint_contacts(analyses, rule)


def rigid_transform():
    theta, phi = 0.9, 0.4
    rotation_z = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]
    )
    return rotation_x @ rotation_z, np.array([5.0, -3.0, 2.0])


class JointContactWindowTests(unittest.TestCase):
    def test_joint_states_require_all_pairs_at_same_frame(self):
        result = run_joint_analysis()
        self.assertEqual([point.state for point in result.points], EXPECTED_JOINT_STATES)

        # frame 0: left-right is in contact but left-top is separated, so the
        # staggered observations must not be stitched into joint contact.
        self.assertEqual(result.points[0].member_states["left__right"], "observed_contact")
        self.assertEqual(result.points[0].member_states["left__top"], "no_contact")
        self.assertEqual(result.points[0].blocking_members[0]["member"], "left__top")

    def test_proven_windows_do_not_cross_brief_separation_without_handoff(self):
        result = run_joint_analysis(max_handoff_gap=0)
        windows = [(w.start_index, w.end_index, w.status) for w in result.windows]
        self.assertEqual(
            windows,
            [(1, 1, "closed"), (3, 3, "interrupted_unknown"), (5, 6, "closed")],
        )
        self.assertEqual(
            [(b["start_index"], b["end_index"]) for b in result.blocking_segments],
            [(0, 0), (2, 2), (7, 7)],
        )
        blocker = result.blocking_segments[1]["frames"][0]["blocking_members"][0]
        self.assertEqual(blocker["member"], "left__right")
        self.assertAlmostEqual(blocker["distance"], 0.9, places=7)
        self.assertEqual(blocker["exit_distance"], EXIT_DISTANCE)

    def test_handoff_gap_bridges_brief_separation_but_marks_it(self):
        result = run_joint_analysis(max_handoff_gap=1)
        windows = [(w.start_index, w.end_index, w.status) for w in result.windows]
        self.assertEqual(windows, [(1, 3, "interrupted_unknown"), (5, 6, "closed")])
        first = result.windows[0]
        self.assertEqual(first.proven_contact_indices, [1, 3])
        self.assertEqual(first.handoff_indices, [2])
        self.assertEqual(first.handoff_details[0]["blocking_members"][0]["member"], "left__right")
        # the trailing separation at frame 7 is not followed by contact, so it
        # still blocks even with an allowed handoff gap
        self.assertEqual(
            [(b["start_index"], b["end_index"]) for b in result.blocking_segments],
            [(0, 0), (7, 7)],
        )

    def test_partial_missing_atoms_only_make_dependent_frames_undetermined(self):
        result = run_joint_analysis()
        self.assertEqual(
            [(u["start_index"], u["end_index"]) for u in result.undetermined_ranges],
            [(4, 4)],
        )
        frame = result.undetermined_ranges[0]["frames"][0]
        self.assertEqual(frame["member_states"]["left__right"], "observed_contact")
        unknown = frame["unknown_members"][0]
        self.assertEqual(unknown["member"], "left__top")
        self.assertEqual(unknown["evidence_state"], "missing_atoms")
        self.assertEqual(unknown["missing_atoms"], {"top": [12]})
        # the window before the unknown frame is interrupted, not lost or active
        self.assertEqual(result.windows[1].status, "interrupted_unknown")
        self.assertEqual(result.windows[1].end_index, 3)

    def test_window_reports_member_support_intervals(self):
        result = run_joint_analysis(max_handoff_gap=1)
        support = result.windows[0].member_support
        self.assertIn("left__right", support)
        self.assertIn("left__top", support)
        right_intervals = support["left__right"]["intervals"]
        self.assertTrue(any(item["start_index"] == 0 and item["status"] == "lost" for item in right_intervals))
        self.assertTrue(any(item["start_index"] == 3 for item in right_intervals))
        top_states = support["left__top"]["frame_states"]
        self.assertEqual(top_states[1], "observed_contact")
        self.assertEqual(top_states[3], "observed_contact")

    def test_rigid_transform_does_not_change_joint_results(self):
        baseline = run_joint_analysis(max_handoff_gap=1)
        moved = run_joint_analysis(transform=rigid_transform(), max_handoff_gap=1)
        self.assertEqual(
            [point.state for point in moved.points],
            [point.state for point in baseline.points],
        )
        self.assertEqual(
            _rounded([w.to_dict() for w in moved.windows]),
            _rounded([w.to_dict() for w in baseline.windows]),
        )
        self.assertEqual(
            _rounded(moved.undetermined_ranges), _rounded(baseline.undetermined_ranges)
        )
        self.assertEqual(_rounded(moved.blocking_segments), _rounded(baseline.blocking_segments))


def _rounded(value):
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


class JointContactScenarioTests(unittest.TestCase):
    def test_scenario_comparison_reports_threshold_and_handoff_effects(self):
        series = make_series()
        scenarios = [
            {"members": [
                {"region_a": "left", "region_b": "right", "enter_distance": ENTER_DISTANCE, "exit_distance": EXIT_DISTANCE},
                {"region_a": "left", "region_b": "top", "enter_distance": ENTER_DISTANCE, "exit_distance": EXIT_DISTANCE},
            ], "max_handoff_gap": 0, "name": "strict"},
            {"members": [
                {"region_a": "left", "region_b": "right", "enter_distance": ENTER_DISTANCE, "exit_distance": EXIT_DISTANCE},
                {"region_a": "left", "region_b": "top", "enter_distance": ENTER_DISTANCE, "exit_distance": EXIT_DISTANCE},
            ], "max_handoff_gap": 1, "name": "handoff-1"},
            {"members": [
                {"region_a": "left", "region_b": "right", "enter_distance": 0.95, "exit_distance": 1.0},
                {"region_a": "left", "region_b": "top", "enter_distance": 0.95, "exit_distance": 1.0},
            ], "max_handoff_gap": 0, "name": "loose"},
        ]
        report = analyze_joint_contact_scenarios(series, scenarios)
        summary = {item["name"]: item for item in report["summary"]}
        self.assertEqual(summary["strict"]["proven_window_count"], 3)
        self.assertEqual(summary["handoff-1"]["proven_window_count"], 2)
        self.assertEqual(summary["handoff-1"]["proven_duration"], 4)
        self.assertEqual(summary["strict"]["undetermined_duration"], 1)
        # a loose threshold treats every measured frame as contact: no blocking,
        # but the missing-atoms frame still splits coverage into two windows
        self.assertEqual(summary["loose"]["proven_window_count"], 2)
        self.assertEqual(summary["loose"]["blocking_duration"], 0)
        self.assertEqual(summary["loose"]["proven_duration"], 7)
        self.assertEqual(summary["loose"]["undetermined_duration"], 1)
        self.assertEqual(len(report["scenarios"]), 3)


class JointContactProjectTests(unittest.TestCase):
    def test_project_run_saves_joint_contact_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = make_reference()
            reference_path = directory / "reference.xyz"
            save_structure(reference, reference_path)
            conformers, correspondences = make_conformers()
            specs = []
            for index, (conformer, correspondence) in enumerate(zip(conformers, correspondences)):
                path = directory / ("frame-%d.xyz" % index)
                save_structure(conformer.structure, path)
                specs.append(ConformerCorrespondenceSpec.from_correspondence(path, correspondence))
            project = ComparisonProject(
                reference_path=str(reference_path),
                conformers=specs,
                reference_regions={"core": [0, 1, 2, 3]},
                regions={
                    "left": (LEFT_KEYS, None),
                    "right": (RIGHT_KEYS, None),
                    "top": (TOP_KEYS, None),
                },
                relative_motion_pairs=[["left", "right"], ["left", "top"]],
                joint_contact_rules=[make_rule(max_handoff_gap=1).to_dict()],
            )
            project_path = directory / "project.json"
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            self.assertEqual(len(reopened.joint_contact_rules), 1)
            self.assertEqual(reopened.joint_contact_rules[0]["max_handoff_gap"], 1)

            run_comparison_project(reopened, report_directory=directory / "reports")
            report_path = (
                directory / "reports" / "core" / "joint_contacts" / "left__right+left__top.json"
            )
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["rule"]["max_handoff_gap"], 1)
            self.assertEqual(
                [(w["start_index"], w["end_index"], w["status"]) for w in payload["windows"]],
                [(1, 3, "interrupted_unknown"), (5, 6, "closed")],
            )
            self.assertEqual(payload["undetermined_duration"], 1)


if __name__ == "__main__":
    unittest.main()
