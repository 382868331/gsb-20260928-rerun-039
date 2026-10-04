import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ComparisonProject,
    ConformerCorrespondenceSpec,
    Structure,
    candidates_from_identity,
    load_comparison_project,
    match_atoms,
    run_comparison_project,
    save_comparison_project,
    save_structure,
    transform_structure,
)


# core atoms 0..3; contact regions A (4..5) and B (6..7)
REFERENCE_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.2],
        [0.2, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 0.2, 0.0],
    ],
    dtype=float,
)

RULE = {
    "region_a": "A",
    "region_b": "B",
    "enter_distance": 0.3,
    "exit_distance": 0.6,
    "min_contact_duration": 1,
}


def structure(coords, serials=None):
    serials = serials if serials is not None else list(range(1, len(coords) + 1))
    return Structure(
        [Atom(index, "C", np.asarray(coords[index], dtype=float), serial=serials[index]) for index in range(len(coords))]
    )


def frame_coords(a_atoms, b_atoms):
    coords = {index: REFERENCE_COORDS[index] for index in range(4)}
    for key, value in zip((4, 5), a_atoms):
        coords[key] = value
    for key, value in zip((6, 7), b_atoms):
        coords[key] = value
    return np.array([coords[index] for index in sorted(coords)])


def rigid_mobile(mobile):
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return transform_structure(mobile, rotation, np.array([1.2, -0.7, 0.4]))


class ContactMissingAtomsTests(unittest.TestCase):
    def _build_frames(self):
        a_near = [np.array([0.0, 0.0, 0.2]), np.array([0.2, 0.0, 0.0])]
        a_recovered = [np.array([0.05, 0.0, 0.2]), np.array([0.25, 0.0, 0.0])]
        b_contact = [np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.0, 0.1])]
        # center distance ~0.351, inside the 0.3/0.6 hysteresis band
        b_hold = [np.array([0.35, 0.0, 0.1]), np.array([0.55, 0.0, 0.15])]
        b_far = [np.array([5.0, 0.0, 0.0]), np.array([5.0, 0.2, 0.0])]
        b_one = [np.array([0.0, 0.0, 0.1])]
        return [
            ("contact", frame_coords(a_near, b_contact), list(range(8))),
            ("hold", frame_coords(a_near, b_hold), list(range(8))),
            ("partial_missing", frame_coords(a_near, b_one), [index for index in range(8) if index != 7]),
            ("all_missing", frame_coords(a_near, []), list(range(6))),
            ("recovery", frame_coords(a_recovered, b_contact), list(range(8))),
            ("separation", frame_coords(a_near, b_far), list(range(8))),
            ("recontact", frame_coords(a_near, b_contact), list(range(8))),
        ]

    def test_lifecycle_keeps_missing_frames_unknown_through_save_and_reopen(self):
        reference = structure(REFERENCE_COORDS)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ref_path = directory / "reference.xyz"
            save_structure(reference, ref_path)
            specs = []
            for name, coords, present in self._build_frames():
                present = sorted(present)
                mobile = rigid_mobile(structure([coords[key] for key in present]))
                path = directory / f"{name}.xyz"
                save_structure(mobile, path)
                pairs = [(ref_key, mobile_key) for mobile_key, ref_key in enumerate(present)]
                correspondence = match_atoms(reference, mobile, explicit_pairs=pairs)
                specs.append(
                    ConformerCorrespondenceSpec.from_correspondence(path, correspondence, name=name)
                )
            project = ComparisonProject(
                reference_path=str(ref_path),
                conformers=specs,
                reference_regions={"core": [0, 1, 2, 3]},
                regions={"A": ([4, 5], None), "B": ([6, 7], None)},
                relative_motion_pairs=[["A", "B"]],
                contact_rules=[RULE],
            )
            save_comparison_project(project, directory / "project.json")
            run_comparison_project(
                load_comparison_project(directory / "project.json"),
                report_directory=directory / "reports",
            )
            contact_path = directory / "reports" / "core" / "contacts" / "A__B__0.3__0.6__min1.json"
            data = json.loads(contact_path.read_text(encoding="utf-8"))

            evidence = [point["evidence_state"] for point in data["points"]]
            states = [point["state"] for point in data["points"]]
            self.assertEqual(
                evidence,
                [
                    "observed_contact",
                    "hysteresis_hold",
                    "missing_partial_atoms",
                    "missing_all_atoms",
                    "observed_contact",
                    "no_contact",
                    "observed_contact",
                ],
            )
            self.assertEqual(states[2], "unresolved")
            self.assertEqual(states[3], "unresolved")
            self.assertIsNone(data["points"][3]["distance"])
            self.assertEqual(
                data["points"][2]["atom_coverage"]["B"],
                {"expected_atoms": 2, "observed_atoms": 1, "missing_atoms": 1},
            )
            self.assertEqual(
                data["points"][3]["atom_coverage"]["B"],
                {"expected_atoms": 2, "observed_atoms": 0, "missing_atoms": 2},
            )
            self.assertEqual(
                [event["event_type"] for event in data["events"]],
                ["formed", "lost", "formed"],
            )
            self.assertEqual(len(data["intervals"]), 2)
            first = data["intervals"][0]
            self.assertEqual(first["observed_contact_indices"], [0, 4])
            self.assertEqual(first["hysteresis_hold_indices"], [1])
            self.assertEqual(first["unresolved_indices"], [2, 3])
            self.assertEqual(first["unresolved_gap_duration"], 2)
            self.assertTrue(first["has_unresolved_gap"])
            self.assertFalse(first["continuous_observed_contact"])
            segment_states = [segment["evidence_state"] for segment in first["evidence_segments"]]
            self.assertIn("missing_partial_atoms", segment_states)
            self.assertIn("missing_all_atoms", segment_states)
            # Atomic evidence stays reviewable even when a region vanished entirely.
            self.assertEqual(data["points"][3]["atom_pairs"]["A"], [[4, 4], [5, 5]])
            self.assertEqual(data["points"][3]["atom_pairs"]["B"], [])
            second = data["intervals"][1]
            self.assertEqual(second["observed_contact_indices"], [6])
            self.assertEqual(second["status"], "active")

    def test_candidate_switch_resolves_disputed_frame_and_stays_reviewable(self):
        # Duplicate serials on region A make its two atoms exchange candidates.
        serials = (1, 2, 3, 4, 9, 9, 7, 8)
        reference = structure(REFERENCE_COORDS, serials=serials)
        swapped = REFERENCE_COORDS.copy()
        swapped[[4, 5]] = swapped[[5, 4]]
        ambiguous = rigid_mobile(structure(swapped, serials=serials))

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            ref_path = directory / "reference.xyz"
            amb_path = directory / "ambiguous.xyz"
            save_structure(reference, ref_path)
            save_structure(ambiguous, amb_path)
            candidates = candidates_from_identity(reference, ambiguous)
            project = ComparisonProject(
                reference_path=str(ref_path),
                conformers=[
                    ConformerCorrespondenceSpec.from_correspondence(
                        amb_path,
                        candidates[1].to_correspondence(),
                        name="ambiguous",
                        candidate_id=candidates[1].candidate_id,
                        alternatives=candidates,
                    ),
                ],
                reference_regions={"core": [0, 1, 2, 3]},
                regions={"A": ([4, 5], None), "B": ([6, 7], None)},
                relative_motion_pairs=[["A", "B"]],
                contact_rules=[RULE],
            )
            save_comparison_project(project, directory / "project.json")
            reopened = load_comparison_project(directory / "project.json")
            run_comparison_project(reopened, report_directory=directory / "reports_before")
            before = json.loads(
                (
                    directory / "reports_before" / "core" / "contacts" / "A__B__0.3__0.6__min1.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(before["points"][0]["atom_pairs"]["A"], [[4, 4], [5, 5]])
            self.assertEqual(before["points"][0]["candidate_id"], candidates[1].candidate_id)

            reopened.conformers[0].candidate_id = candidates[0].candidate_id
            save_comparison_project(reopened, directory / "project_decided.json")
            decided = load_comparison_project(directory / "project_decided.json")
            run_comparison_project(decided, report_directory=directory / "reports_after")
            after = json.loads(
                (
                    directory / "reports_after" / "core" / "contacts" / "A__B__0.3__0.6__min1.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(after["points"][0]["candidate_id"], candidates[0].candidate_id)
            self.assertEqual(after["points"][0]["atom_pairs"]["A"], [[4, 5], [5, 4]])
            # Switching equivalent atoms changes the recorded atom basis but a
            # global rigid transform leaves the contact distance unchanged.
            self.assertEqual(
                before["points"][0]["distance"], after["points"][0]["distance"]
            )
            self.assertEqual(len(after["points"][0]["alternative_measurements"]), 2)


if __name__ == "__main__":
    unittest.main()
