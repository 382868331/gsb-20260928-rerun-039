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
    Structure,
    analyze_contacts,
    analyze_relative_motion,
    candidates_from_identity,
    compare_ensemble,
    confirm_project_correspondence,
    load_comparison_project,
    run_comparison_project,
    save_comparison_project,
    save_structure,
    transform_structure,
    withdraw_project_correspondence,
)

from molecular_toolkit import RelativeMotionPoint, RelativeMotionSeries, save_contact_analysis

ENTER_DISTANCE = 0.3
EXIT_DISTANCE = 0.6
CORE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
LEFT_OFFSETS = [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
LEFT_BASE = (3.0, 0.0, 0.0)
RIGHT_BASE = (4.0, 0.0, 0.0)
LEFT_KEYS = [4, 5, 6]
RIGHT_KEYS = [7, 8, 9]
SERIALS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 50, 5: 50, 6: 7, 7: 8, 8: 9, 9: 10}
FRAME_DISTANCES = [0.2, 0.45, None, None, 0.25, 0.9, 0.15]
FRAME_RIGHT_KEYS = [
    RIGHT_KEYS,
    RIGHT_KEYS,
    [],
    [7, 8],
    RIGHT_KEYS,
    RIGHT_KEYS,
    RIGHT_KEYS,
]
EXPECTED_STATES = [
    "contact",
    "contact_held",
    "unresolved",
    "unresolved",
    "contact",
    "no_contact",
    "contact",
]


def _xyz(base, offsets):
    return [tuple(np.asarray(base) + np.asarray(offset)) for offset in offsets]


def make_reference():
    coordinates = CORE + _xyz(LEFT_BASE, LEFT_OFFSETS) + _xyz(RIGHT_BASE, LEFT_OFFSETS)
    return Structure(
        [Atom(key, "C", coordinates[key], serial=SERIALS[key]) for key in range(10)]
    )


def make_mobile(distance, right_keys):
    coordinates = list(CORE) + _xyz(LEFT_BASE, LEFT_OFFSETS)
    keys = list(range(7))
    if right_keys:
        right_base = np.asarray(LEFT_BASE) + np.asarray((distance, 0.0, 0.0))
        right_xyz = _xyz(tuple(right_base), LEFT_OFFSETS)
        for position, key in enumerate(right_keys):
            coordinates.append(right_xyz[RIGHT_KEYS.index(key)])
            keys.append(key)
    atoms = [Atom(index, "C", coordinates[index], serial=SERIALS[keys[index]]) for index in range(len(keys))]
    return Structure(atoms), keys


def make_correspondence(keys):
    present = set(keys)
    return AtomCorrespondence(
        pairs=[(key, index) for index, key in enumerate(keys)],
        unmatched_reference=[key for key in range(10) if key not in present],
        unmatched_mobile=[],
        ambiguous_identities={},
    )


def make_conformers(transform=None):
    conformers = []
    correspondences = []
    for index, (distance, right_keys) in enumerate(zip(FRAME_DISTANCES, FRAME_RIGHT_KEYS)):
        mobile, keys = make_mobile(distance if distance is not None else 0.2, right_keys)
        if transform is not None:
            rotation, translation = transform
            mobile = transform_structure(mobile, rotation, translation)
        conformers.append(Conformer(name=f"frame-{index}", structure=mobile))
        correspondences.append(make_correspondence(keys))
    return conformers, correspondences


def run_contact_analysis(transform=None):
    reference = make_reference()
    conformers, correspondences = make_conformers(transform)
    comparisons = compare_ensemble(
        reference,
        list(zip(conformers, correspondences)),
        [0, 1, 2, 3],
        {"left": (LEFT_KEYS, None), "right": (RIGHT_KEYS, None)},
    )
    series = analyze_relative_motion(comparisons, "left", "right")
    rule = ContactRule(
        region_a="left",
        region_b="right",
        enter_distance=ENTER_DISTANCE,
        exit_distance=EXIT_DISTANCE,
        min_contact_duration=1,
    )
    return analyze_contacts(series, rule), series


def rigid_transform():
    theta, phi = 0.9, 0.4
    rotation_z = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]
    )
    return rotation_x @ rotation_z, np.array([5.0, -3.0, 2.0])


class MissingAtomContactTests(unittest.TestCase):
    def test_missing_frames_stay_unknown_and_analysis_continues(self):
        analysis, series = run_contact_analysis()

        self.assertEqual([point.state for point in analysis.points], EXPECTED_STATES)
        self.assertEqual(
            [point.evidence_state for point in analysis.points],
            [
                "observed_contact",
                "hysteresis_hold",
                "missing_atoms",
                "missing_atoms",
                "observed_contact",
                "no_contact",
                "observed_contact",
            ],
        )

        fully_missing = series.points[2]
        self.assertIsNone(fully_missing.distance)
        self.assertEqual(fully_missing.missing_atoms, {"right": [7, 8, 9]})
        self.assertEqual(fully_missing.atom_pairs["right"], [])
        self.assertIn("no paired atoms", analysis.points[2].explanation)

        partially_missing = series.points[3]
        self.assertIsNone(partially_missing.distance)
        self.assertEqual(partially_missing.missing_atoms, {"right": [9]})
        self.assertEqual(partially_missing.atom_pairs["right"], [[7, 7], [8, 8]])
        self.assertIn("partially measured", analysis.points[3].explanation)
        self.assertIn("neither a zero-distance contact nor a confirmed separation", analysis.points[3].explanation)

        events = [(event.event_type, event.index, event.contact_start_index) for event in analysis.events]
        self.assertEqual(events, [("formed", 0, 0), ("lost", 5, 0), ("formed", 6, 6)])

        self.assertEqual(len(analysis.intervals), 2)
        first = analysis.intervals[0].to_dict()
        self.assertEqual(first["start_index"], 0)
        self.assertEqual(first["end_index"], 5)
        self.assertEqual(first["status"], "lost")
        self.assertEqual(first["observed_contact_indices"], [0, 4])
        self.assertEqual(first["hysteresis_hold_indices"], [1])
        self.assertEqual(first["unresolved_gap_duration"], 2)
        self.assertTrue(first["has_unresolved_gap"])
        self.assertFalse(first["continuous_observed_contact"])
        self.assertIn(
            {"start_index": 2, "end_index": 3, "evidence_state": "missing_atoms"},
            first["evidence_segments"],
        )

        second = analysis.intervals[1].to_dict()
        self.assertEqual(second["start_index"], 6)
        self.assertEqual(second["status"], "active")
        self.assertEqual(second["observed_contact_indices"], [6])

    def test_rigid_transform_does_not_change_conclusions(self):
        baseline, baseline_series = run_contact_analysis()
        moved, moved_series = run_contact_analysis(transform=rigid_transform())

        self.assertEqual(
            [point.state for point in moved.points],
            [point.state for point in baseline.points],
        )
        self.assertEqual(
            [(event.event_type, event.index) for event in moved.events],
            [(event.event_type, event.index) for event in baseline.events],
        )
        for expected, actual in zip(baseline_series.points, moved_series.points):
            self.assertEqual(actual.missing_atoms, expected.missing_atoms)
            if expected.distance is None:
                self.assertIsNone(actual.distance)
            else:
                self.assertAlmostEqual(actual.distance, expected.distance, places=7)


class MissingAtomProjectTests(unittest.TestCase):
    def _build_project(self, directory):
        directory = Path(directory)
        reference = make_reference()
        reference_path = directory / "reference.json"
        save_structure(reference, reference_path)
        specs = []
        for index, (distance, right_keys) in enumerate(zip(FRAME_DISTANCES, FRAME_RIGHT_KEYS)):
            mobile, _ = make_mobile(distance if distance is not None else 0.2, right_keys)
            path = directory / f"frame-{index}.json"
            save_structure(mobile, path)
            candidates = candidates_from_identity(reference, mobile)
            self.assertEqual(len(candidates), 2)
            specs.append(
                ConformerCorrespondenceSpec.from_correspondence(
                    path,
                    candidates[0].to_correspondence(),
                    name=f"frame-{index}",
                    candidate_id=candidates[0].candidate_id,
                    alternatives=candidates,
                )
            )
        return ComparisonProject(
            reference_path=str(reference_path),
            conformers=specs,
            reference_regions={"core": [0, 1, 2, 3]},
            regions={"left": (LEFT_KEYS, None), "right": (RIGHT_KEYS, None)},
            relative_motion_pairs=[["left", "right"]],
            contact_rules=[
                {
                    "region_a": "left",
                    "region_b": "right",
                    "enter_distance": ENTER_DISTANCE,
                    "exit_distance": EXIT_DISTANCE,
                    "min_contact_duration": 1,
                }
            ],
        )

    @staticmethod
    def _contacts_report(directory):
        path = (
            Path(directory)
            / "core"
            / "contacts"
            / f"left__right__{ENTER_DISTANCE:g}__{EXIT_DISTANCE:g}__min1.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_save_reopen_and_candidate_switch_reproduce_missing_aware_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self._build_project(directory)
            run_comparison_project(project, report_directory=Path(directory) / "reports-initial")
            initial = self._contacts_report(Path(directory) / "reports-initial")

            self.assertEqual([point["state"] for point in initial["points"]], EXPECTED_STATES)
            self.assertEqual(initial["points"][2]["missing_atoms"], {"right": [7, 8, 9]})
            self.assertEqual(initial["points"][3]["missing_atoms"], {"right": [9]})
            self.assertEqual(initial["points"][3]["atom_pairs"]["right"], [[7, 7], [8, 8]])
            self.assertEqual(initial["intervals"][0]["unresolved_gap_duration"], 2)
            self.assertEqual(
                [(event["event_type"], event["index"]) for event in initial["events"]],
                [("formed", 0), ("lost", 5), ("formed", 6)],
            )

            project_path = Path(directory) / "project.json"
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            run_comparison_project(reopened, report_directory=Path(directory) / "reports-reopened")
            reopened_report = self._contacts_report(Path(directory) / "reports-reopened")
            self.assertEqual(reopened_report, initial)

            decision = confirm_project_correspondence(
                project, 0, len(FRAME_DISTANCES) - 1, candidate_id="cand-001"
            )
            run_comparison_project(project, report_directory=Path(directory) / "reports-switched")
            switched = self._contacts_report(Path(directory) / "reports-switched")
            self.assertEqual([point["state"] for point in switched["points"]], EXPECTED_STATES)
            self.assertEqual(
                [point["candidate_id"] for point in switched["points"]],
                ["cand-001"] * len(FRAME_DISTANCES),
            )
            self.assertEqual(
                sorted(map(tuple, switched["points"][0]["atom_pairs"]["left"])),
                [(4, 5), (5, 4), (6, 6)],
            )
            self.assertEqual(switched["points"][2]["missing_atoms"], {"right": [7, 8, 9]})
            for switched_interval, initial_interval in zip(switched["intervals"], initial["intervals"]):
                for key in (
                    "start_index",
                    "end_index",
                    "status",
                    "observed_contact_duration",
                    "hysteresis_hold_duration",
                    "unresolved_gap_duration",
                ):
                    self.assertEqual(switched_interval[key], initial_interval[key])
                self.assertEqual(
                    [s for s in switched_interval["evidence_segments"] if s["evidence_state"] == "missing_atoms"],
                    [s for s in initial_interval["evidence_segments"] if s["evidence_state"] == "missing_atoms"],
                )
            self.assertEqual(switched["intervals"][0]["confirmed_contact_duration"], 2)
            self.assertEqual(switched["events"], initial["events"])

            withdraw_project_correspondence(project, decision["decision_id"])
            run_comparison_project(project, report_directory=Path(directory) / "reports-withdrawn")
            withdrawn = self._contacts_report(Path(directory) / "reports-withdrawn")
            self.assertEqual(
                [point["state"] for point in withdrawn["points"]],
                [point["state"] for point in initial["points"]],
            )
            self.assertEqual(
                [point["distance"] for point in withdrawn["points"]],
                [point["distance"] for point in initial["points"]],
            )
            self.assertEqual(withdrawn["intervals"], initial["intervals"])


def make_motion_point(index, distance, missing_atoms=None, atom_pairs=None):
    return RelativeMotionPoint(
        index=index,
        conformer_name=f"frame-{index}",
        source_path=None,
        region_a="left",
        region_b="right",
        reference_region=None,
        distance=distance,
        previous_distance=None,
        distance_change=None,
        motion_state=None,
        angle_deg=None,
        angle_status="ok",
        distance_range=None,
        angle_range=None,
        region_a_center=None,
        region_b_center=None,
        region_a_axis=None,
        region_b_axis=None,
        region_a_translation_distance=None,
        region_b_translation_distance=None,
        region_a_rotation_angle=None,
        region_b_rotation_angle=None,
        region_a_internal_rmsd=None,
        region_b_internal_rmsd=None,
        atom_pairs=atom_pairs if atom_pairs is not None else {"left": [[4, 4]], "right": [[7, 7]]},
        candidate_id=None,
        correspondence_status="ok",
        available_interpretations=1,
        missing_atoms=missing_atoms or {},
    )


def make_series(frame_distances, missing_atoms_by_index=None, atom_pairs_by_index=None):
    missing_atoms_by_index = missing_atoms_by_index or {}
    atom_pairs_by_index = atom_pairs_by_index or {}
    points = [
        make_motion_point(
            index,
            distance,
            missing_atoms=missing_atoms_by_index.get(index),
            atom_pairs=atom_pairs_by_index.get(index),
        )
        for index, distance in enumerate(frame_distances)
    ]
    return RelativeMotionSeries(
        region_a="left", region_b="right", reference_region=None, points=points
    )


TRAILING_RULE = ContactRule(
    region_a="left",
    region_b="right",
    enter_distance=ENTER_DISTANCE,
    exit_distance=EXIT_DISTANCE,
    min_contact_duration=1,
)


class TrailingMissingEvidenceTests(unittest.TestCase):
    def test_contact_then_fully_missing_tail_is_not_reported_active(self):
        series = make_series(
            [0.2, 0.25, None, None],
            missing_atoms_by_index={2: {"right": [7, 8, 9]}, 3: {"right": [7, 8, 9]}},
            atom_pairs_by_index={
                2: {"left": [[4, 4]], "right": []},
                3: {"left": [[4, 4]], "right": []},
            },
        )
        analysis = analyze_contacts(series, TRAILING_RULE)

        self.assertEqual(
            [point.state for point in analysis.points],
            ["contact", "contact", "unresolved", "unresolved"],
        )
        self.assertEqual(len(analysis.intervals), 1)
        interval = analysis.intervals[0].to_dict()
        self.assertEqual(interval["status"], "unresolved")
        self.assertTrue(interval["ends_with_unresolved_evidence"])
        self.assertEqual(
            interval["trailing_unresolved_segment"],
            {"start_index": 2, "end_index": 3, "evidence_state": "missing_atoms"},
        )
        self.assertEqual(interval["last_observed_contact_index"], 1)
        self.assertIsNone(interval["last_confirmed_contact_index"])
        self.assertEqual(interval["sustained_duration"], 2)
        self.assertEqual(interval["unresolved_gap_duration"], 2)
        self.assertIsNone(interval["end_index"])
        self.assertEqual(
            [(event.event_type, event.index) for event in analysis.events],
            [("formed", 0)],
        )

    def test_contact_then_partially_missing_tail_keeps_retained_pairs(self):
        series = make_series(
            [0.2, None],
            missing_atoms_by_index={1: {"right": [9]}},
            atom_pairs_by_index={1: {"left": [[4, 4]], "right": [[7, 7], [8, 8]]}},
        )
        analysis = analyze_contacts(series, TRAILING_RULE)

        self.assertEqual(analysis.points[1].state, "unresolved")
        self.assertIn("partially measured", analysis.points[1].explanation)
        interval = analysis.intervals[0].to_dict()
        self.assertEqual(interval["status"], "unresolved")
        self.assertTrue(interval["ends_with_unresolved_evidence"])
        self.assertEqual(
            interval["trailing_unresolved_segment"],
            {"start_index": 1, "end_index": 1, "evidence_state": "missing_atoms"},
        )
        self.assertEqual(interval["last_observed_contact_index"], 0)
        self.assertEqual(interval["sustained_duration"], 1)

    def test_missing_tail_recovered_by_hysteresis_hold_stays_active(self):
        series = make_series(
            [0.2, None, 0.45],
            missing_atoms_by_index={1: {"right": [7, 8, 9]}},
            atom_pairs_by_index={1: {"left": [[4, 4]], "right": []}},
        )
        analysis = analyze_contacts(series, TRAILING_RULE)

        self.assertEqual(
            [point.state for point in analysis.points],
            ["contact", "unresolved", "contact_held"],
        )
        interval = analysis.intervals[0].to_dict()
        self.assertEqual(interval["status"], "active")
        self.assertFalse(interval["ends_with_unresolved_evidence"])
        self.assertIsNone(interval["trailing_unresolved_segment"])
        self.assertEqual(interval["hysteresis_hold_indices"], [2])
        self.assertEqual(interval["last_observed_contact_index"], 0)
        self.assertEqual(interval["unresolved_gap_duration"], 1)

    def test_recovery_then_release_beyond_exit_threshold_closes_interval(self):
        series = make_series(
            [0.2, None, 0.45, 0.9],
            missing_atoms_by_index={1: {"right": [7, 8, 9]}},
            atom_pairs_by_index={1: {"left": [[4, 4]], "right": []}},
        )
        analysis = analyze_contacts(series, TRAILING_RULE)

        self.assertEqual(
            [point.state for point in analysis.points],
            ["contact", "unresolved", "contact_held", "no_contact"],
        )
        interval = analysis.intervals[0].to_dict()
        self.assertEqual(interval["status"], "lost")
        self.assertEqual(interval["end_index"], 3)
        self.assertFalse(interval["ends_with_unresolved_evidence"])
        self.assertIsNone(interval["trailing_unresolved_segment"])
        self.assertEqual(
            [(event.event_type, event.index) for event in analysis.events],
            [("formed", 0), ("lost", 3)],
        )

    def test_trailing_missing_summary_survives_export_and_reopen(self):
        series = make_series(
            [0.2, 0.25, None],
            missing_atoms_by_index={2: {"right": [7, 8, 9]}},
            atom_pairs_by_index={2: {"left": [[4, 4]], "right": []}},
        )
        analysis = analyze_contacts(series, TRAILING_RULE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contacts.json"
            save_contact_analysis(analysis, path)
            exported = json.loads(path.read_text(encoding="utf-8"))
            reopened = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(reopened, exported)
        interval = exported["intervals"][0]
        self.assertEqual(interval["status"], "unresolved")
        self.assertTrue(interval["ends_with_unresolved_evidence"])
        self.assertEqual(
            interval["trailing_unresolved_segment"],
            {"start_index": 2, "end_index": 2, "evidence_state": "missing_atoms"},
        )
        self.assertEqual(interval["last_observed_contact_index"], 1)
        self.assertEqual(interval["sustained_duration"], 2)
        self.assertEqual(interval["unresolved_gap_duration"], 1)


if __name__ == "__main__":
    unittest.main()
