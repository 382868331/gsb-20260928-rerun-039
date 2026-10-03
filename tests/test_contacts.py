import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ComparisonProject,
    Conformer,
    ConformerCorrespondenceSpec,
    ContactThresholds,
    CorrespondenceCandidate,
    Structure,
    analyze_contact_persistence,
    compare_ensemble,
    confirm_project_correspondence,
    load_comparison_project,
    preview_project_contact_decision,
    run_comparison_project,
    save_comparison_project,
    save_structure,
    withdraw_project_correspondence,
)
from molecular_toolkit.correspondence import AtomCorrespondence


CORE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ]
)


def make_reference():
    coords = np.vstack(
        [
            CORE,
            np.array([[2.0, 0.0, 0.0], [2.0, 10.0, 0.0]]),
            np.array([[4.0, 0.0, 0.0], [4.0, 2.0, 0.0]]),
        ]
    )
    return Structure([Atom(i, "C", coords[i], serial=i + 1) for i in range(8)])


def conformer_with_gap(gap: float, name: str):
    coords = np.vstack(
        [
            CORE,
            np.array([[2.0, 0.0, 0.0], [2.0, 10.0, 0.0]]),
            np.array([[2.0 + gap, 0.0, 0.0], [2.0 + gap, 2.0, 0.0]]),
        ]
    )
    return Conformer(name, Structure([Atom(i, "C", coords[i], serial=i + 1) for i in range(8)]))


REGIONS = {"left": ([4, 5], None), "right": ([6, 7], None)}
THRESHOLDS = ContactThresholds(0.5, 0.8)


class HysteresisTests(unittest.TestCase):
    def test_threshold_band_does_not_generate_event_flood(self):
        reference = make_reference()
        gaps = [0.4, 0.65, 0.4, 0.65, 2.0, 0.4]
        conformers = [conformer_with_gap(gap, f"c{i}") for i, gap in enumerate(gaps)]
        correspondences = [
            AtomCorrespondence(pairs=[(i, i) for i in range(8)]) for _ in gaps
        ]
        comparisons = compare_ensemble(
            reference,
            list(zip(conformers, correspondences)),
            stable_reference_keys=[0, 1, 2, 3],
            regions=REGIONS,
        )

        trace = analyze_contact_persistence(
            comparisons, "left", "right", THRESHOLDS, region_definitions=REGIONS
        )

        kinds = [event["event"] for event in trace.events]
        self.assertEqual(kinds, ["contact_formed", "contact_lost", "contact_formed"])
        states = [point.state for point in trace.points]
        self.assertEqual(states, ["contact"] * 4 + ["separated", "contact"])
        self.assertEqual(len(trace.episodes), 2)
        self.assertEqual(trace.episodes[0].point_indices, [0, 1, 2, 3])
        self.assertEqual(trace.episodes[0].end_reason, "separated")

    def test_invalid_hysteresis_band_rejected(self):
        with self.assertRaises(ValueError):
            ContactThresholds(1.0, 0.5)


class MissingAndDisputeTests(unittest.TestCase):
    def test_missing_atoms_split_episodes_and_are_not_loss(self):
        reference = make_reference()
        near = conformer_with_gap(0.4, "near-1")
        far_coords = np.vstack([CORE, np.array([[2.0, 0.0, 0.0]])])
        missing = Conformer(
            "missing",
            Structure([Atom(i, "C", far_coords[i], serial=i + 1) for i in range(5)]),
        )
        near_2 = conformer_with_gap(0.4, "near-2")
        near_pairs = AtomCorrespondence(pairs=[(i, i) for i in range(8)])
        missing_pairs = AtomCorrespondence(pairs=[(i, i) for i in range(4)])

        comparisons = compare_ensemble(
            reference,
            list(zip([near, missing, near_2], [near_pairs, missing_pairs, near_pairs])),
            stable_reference_keys=[0, 1, 2, 3],
            regions=REGIONS,
        )
        trace = analyze_contact_persistence(
            comparisons, "left", "right", THRESHOLDS, region_definitions=REGIONS
        )

        self.assertEqual(
            [point.state for point in trace.points], ["contact", "missing", "contact"]
        )
        middle = trace.points[1]
        self.assertIsNone(middle.distance)
        self.assertEqual(middle.missing_reference_keys["left"], [4, 5])
        self.assertEqual(middle.missing_reference_keys["right"], [6, 7])
        self.assertEqual(
            [event["event"] for event in trace.events],
            ["contact_formed", "contact_interrupted", "contact_formed"],
        )
        self.assertEqual(len(trace.episodes), 2)
        self.assertEqual(trace.episodes[0].end_reason, "missing")
        self.assertEqual([episode.point_indices for episode in trace.episodes], [[0], [2]])

    def test_disagreeing_candidates_are_disputed_not_released(self):
        reference = make_reference()
        coords = np.vstack(
            [
                CORE,
                np.array([[2.4, 0.0, 0.0], [8.2, 0.0, 0.0]]),
                np.array([[2.8, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            ]
        )
        ambiguous = Conformer("ambiguous", Structure([Atom(i, "C", coords[i], serial=i + 1) for i in range(8)]))
        contact_candidate = CorrespondenceCandidate(
            candidate_id="near", pairs=[(i, i) for i in range(8)]
        )
        far_candidate = CorrespondenceCandidate(
            candidate_id="far",
            pairs=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 6), (7, 7)],
            unmatched_reference=[5],
        )
        from molecular_toolkit import compare_conformer

        comparison = compare_conformer(
            reference,
            ambiguous,
            contact_candidate,
            stable_reference_keys=[0, 1, 2, 3],
            regions=REGIONS,
            alternative_candidates=[contact_candidate, far_candidate],
        )
        trace = analyze_contact_persistence(
            [comparison],
            "left",
            "right",
            THRESHOLDS,
            reference=reference,
            stable_reference_keys=[0, 1, 2, 3],
            region_definitions=REGIONS,
        )

        self.assertEqual(trace.points[0].state, "disputed")
        supports = {
            item["candidate_id"]: item["contact_support"]
            for item in trace.points[0].interpretations
        }
        self.assertEqual(supports, {"near": "contact", "far": "no_contact"})
        self.assertEqual(trace.events, [])
        self.assertEqual(trace.episodes, [])

    def test_every_contact_point_links_to_actual_atoms(self):
        reference = make_reference()
        conformer = conformer_with_gap(0.4, "near")
        comparisons = compare_ensemble(
            reference,
            [(conformer, AtomCorrespondence(pairs=[(i, i) for i in range(8)]))],
            stable_reference_keys=[0, 1, 2, 3],
            regions=REGIONS,
        )
        trace = analyze_contact_persistence(
            comparisons, "left", "right", THRESHOLDS, region_definitions=REGIONS
        )
        point = trace.points[0]
        self.assertEqual(point.closest_pair, [4, 6])
        self.assertAlmostEqual(point.distance, 0.4, places=9)
        self.assertEqual(point.atom_pairs["left"], [[4, 4], [5, 5]])
        self.assertEqual(point.atom_pairs["right"], [[6, 6], [7, 7]])
        event = trace.events[0]
        self.assertEqual(event["closest_pair"], [4, 6])
        self.assertEqual(event["source_path"], point.source_path)


class ProjectContactTests(unittest.TestCase):
    def test_contact_trace_is_identical_across_stable_references(self):
        reference = make_reference()
        conformers = [conformer_with_gap(gap, f"c{i}") for i, gap in enumerate([0.4, 2.0])]
        correspondence = AtomCorrespondence(pairs=[(i, i) for i in range(8)])
        paired = list(zip(conformers, [correspondence, correspondence]))
        core_runs = compare_ensemble(
            reference, paired, stable_reference_keys=[0, 1, 2, 3], regions=REGIONS
        )
        alt_runs = compare_ensemble(
            reference, paired, stable_reference_keys=[0, 1, 3, 5], regions=REGIONS
        )
        core_trace = analyze_contact_persistence(
            core_runs, "left", "right", THRESHOLDS, region_definitions=REGIONS
        )
        alt_trace = analyze_contact_persistence(
            alt_runs, "left", "right", THRESHOLDS, region_definitions=REGIONS
        )
        self.assertEqual(
            [p.state for p in core_trace.points], [p.state for p in alt_trace.points]
        )
        self.assertEqual(
            [p.distance for p in core_trace.points], [p.distance for p in alt_trace.points]
        )

    def _ambiguous_project(self, directory: Path) -> tuple[Path, Path, Path]:
        reference = make_reference()
        coords = np.vstack(
            [
                CORE,
                np.array([[2.4, 0.0, 0.0], [8.2, 0.0, 0.0]]),
                np.array([[2.8, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            ]
        )
        ambiguous = Structure([Atom(i, "C", coords[i], serial=i + 1) for i in range(8)])
        clear = Structure(
            [
                Atom(i, "C", conformer_with_gap(0.4, "x").structure.atom(i).xyz, serial=i + 1)
                for i in range(8)
            ]
        )
        ref_path = directory / "reference.xyz"
        amb_path = directory / "ambiguous.xyz"
        clear_path = directory / "clear.xyz"
        save_structure(reference, ref_path)
        save_structure(ambiguous, amb_path)
        save_structure(clear, clear_path)
        candidates = [
            CorrespondenceCandidate(candidate_id="near", pairs=[(i, i) for i in range(8)]),
            CorrespondenceCandidate(
                candidate_id="far",
                pairs=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 6), (7, 7)],
                unmatched_reference=[5],
            ),
        ]
        project = ComparisonProject(
            reference_path=str(ref_path),
            conformers=[
                ConformerCorrespondenceSpec.from_correspondence(
                    amb_path, candidates[0].to_correspondence(),
                    name="ambiguous", alternatives=candidates,
                ),
                ConformerCorrespondenceSpec.from_correspondence(
                    clear_path, AtomCorrespondence(pairs=[(i, i) for i in range(8)]),
                    name="clear",
                ),
            ],
            reference_regions={"core": [0, 1, 2, 3]},
            regions={"left": ([4, 5], None), "right": ([6, 7], None)},
            contact_pairs=[
                {"regions": ["left", "right"], "enter_distance": 0.5, "exit_distance": 0.8}
            ],
        )
        return project, amb_path

    def test_preview_confirm_withdraw_and_report_traceability(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            project, amb_path = self._ambiguous_project(directory)

            preview = preview_project_contact_decision(project, 0, 0, candidate_id="near")
            key = "left__right"
            self.assertEqual(preview[key]["before"]["state_sequence"], ["disputed", "contact"])
            self.assertEqual(preview[key]["after"]["state_sequence"], ["contact", "contact"])
            self.assertEqual(preview[key]["changed_point_indices"], [0])
            self.assertEqual(preview[key]["events_added"][0]["event"], "contact_formed")
            self.assertIsNone(project.conformers[0].candidate_id)

            decision = confirm_project_correspondence(project, 0, 0, candidate_id="near")
            run_comparison_project(project, report_directory=directory / "reports")
            report = json.loads(
                (directory / "reports" / "contacts" / "left__right.json").read_text(encoding="utf-8")
            )
            self.assertEqual([p["state"] for p in report["points"]], ["contact", "contact"])
            self.assertEqual(report["episodes"][0]["observed_frame_count"], 1)
            self.assertEqual(report["episodes"][0]["confirmed_frame_count"], 1)
            self.assertEqual(report["points"][0]["decision_id"], decision["decision_id"])
            self.assertEqual(report["points"][0]["closest_pair"], [4, 6])

            project_path = directory / "project.json"
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            withdrawal = withdraw_project_correspondence(reopened, decision["decision_id"])
            self.assertEqual(withdrawal["status"], "withdrawn")
            self.assertIsNone(reopened.conformers[0].candidate_id)
            self.assertFalse(reopened.conformers[0].confirmed)
            from molecular_toolkit import analyze_project_contacts

            reopened_traces = analyze_project_contacts(reopened)
            self.assertEqual(
                [point.state for point in reopened_traces["left__right"].points],
                ["disputed", "contact"],
            )
            self.assertTrue(
                any(item["status"] == "withdrawn" for item in reopened.correspondence_decisions)
            )
            self.assertTrue(
                any(
                    item["decision_id"] == decision["decision_id"]
                    and item["status"] == "withdrawn"
                    for item in reopened.correspondence_decisions
                )
            )


if __name__ == "__main__":
    unittest.main()
