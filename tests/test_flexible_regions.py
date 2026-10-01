import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ComparisonProject,
    CorrespondenceCandidate,
    ConformerCorrespondenceSpec,
    ReflectionAlignmentError,
    Structure,
    align_on_stable_atoms,
    candidates_from_identity,
    check_mirror,
    compare_ensemble,
    compare_conformer,
    Conformer,
    confirm_candidate,
    confirm_sequence_correspondence,
    decision_affected_measurements,
    enumerate_candidates,
    load_structure,
    load_comparison_project,
    lock_candidate_pairs,
    match_atoms,
    run_comparison_project,
    run_relative_motion_project,
    save_comparison_report,
    save_comparison_project,
    withdraw_correspondence_decision,
)


def tetra_frame(offset=(0.0, 0.0, 0.0), serials=None):
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
        ]
    ) + np.asarray(offset, dtype=float)
    serials = serials or (1, 2, 3, 4, 5, 6)
    return Structure([Atom(i, "C", coordinates[i], serial=serials[i]) for i in range(6)])


class RegionTests(unittest.TestCase):
    def test_stable_fit_reports_local_flexible_region(self):
        reference = tetra_frame()
        rotation = np.array(
            [
                [np.cos(0.4), -np.sin(0.4), 0.0],
                [np.sin(0.4), np.cos(0.4), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mobile_coords = reference.coordinates @ rotation.T + [1.5, -2.0, 0.5]
        # Two non-stable atoms move locally after the rigid-body transform.
        mobile_coords[4] += [0.0, 0.0, 0.8]
        mobile_coords[5] += [0.0, 0.0, -0.4]
        mobile = Structure([Atom(i, "C", mobile_coords[i], serial=i + 1) for i in range(6)])
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(6)]).pairs

        result = align_on_stable_atoms(
            reference,
            mobile,
            pairs,
            stable_reference_keys=[0, 1, 2, 3],
            regions={"flexible": ([4, 5], [4, 5])},
        )

        self.assertAlmostEqual(result.stable_rmsd, 0.0, places=9)
        flexible = result.region("flexible")
        np.testing.assert_allclose(flexible.distances, [0.8, 0.4], atol=1e-9)
        self.assertGreater(flexible.rmsd, result.stable_rmsd)
        self.assertNotIn("unselected", result.regions)

    def test_unselected_pairs_are_reported_automatically(self):
        reference = tetra_frame()
        offsets = [[0.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 0.8], [0.0, 0.0, 1.0]]
        mobile = Structure(
            [Atom(i, "C", reference.atom(i).xyz + offsets[i], serial=i + 1) for i in range(6)]
        )
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(6)]).pairs
        result = align_on_stable_atoms(reference, mobile, pairs, [0, 1, 2, 3])
        self.assertEqual(result.region("unselected").reference_keys, [4, 5])
        np.testing.assert_allclose(result.region("unselected").distances, [0.8, 1.0], atol=1e-9)


class CandidateTests(unittest.TestCase):
    def test_duplicate_identity_candidates_and_manual_lock(self):
        reference = tetra_frame(serials=(1, 2, 3, 4, 9, 9))
        swapped = reference.coordinates.copy()
        swapped[[4, 5]] = swapped[[5, 4]]
        mobile = Structure(
            [Atom(i, "C", swapped[i], serial=(1, 2, 3, 4, 9, 9)[i]) for i in range(6)]
        )

        candidates = candidates_from_identity(reference, mobile)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].pairs[-2:], [(4, 5), (5, 4)])
        self.assertAlmostEqual(candidates[0].rmsd_proper, 0.0, places=9)
        self.assertGreater(candidates[1].rmsd_proper, candidates[0].rmsd_proper)

        locked = lock_candidate_pairs(candidates, [(4, 4)])
        self.assertEqual(len(locked), 1)
        self.assertIn((4, 4), locked[0].locked_pairs)

    def test_missing_atoms_are_not_invented_by_enumeration(self):
        reference = tetra_frame(serials=(1, 1, 3, 4, 5, 6))
        mobile = tetra_frame(offset=(2.0, 0.0, 0.0), serials=(1, 1, 3, 4, 5, 7))
        correspondence = match_atoms(reference, mobile)
        candidates = enumerate_candidates(
            reference,
            mobile,
            [( [0, 1], [0, 1] )],
            base_pairs=correspondence.pairs,
        )
        for candidate in candidates:
            paired_refs = {ref for ref, _ in candidate.pairs}
            paired_mobs = {mob for _, mob in candidate.pairs}
            self.assertNotIn(5, paired_mobs)
            self.assertEqual(len(paired_refs), len(candidate.pairs))

    def test_mirror_candidate_is_flagged_and_not_accepted_silently(self):
        reference = tetra_frame()
        reflection = np.diag([1.0, 1.0, -1.0])
        mobile = Structure(
            [Atom(i, "C", reflection @ reference.atom(i).xyz, serial=i + 1) for i in range(6)]
        )
        candidates = candidates_from_identity(reference, mobile)
        self.assertTrue(candidates[0].is_mirror_candidate)
        self.assertLess(candidates[0].rmsd_reflected, candidates[0].rmsd_proper)

        pairs = [(i, i) for i in range(6)]
        with self.assertRaises(ReflectionAlignmentError):
            check_mirror(reference, mobile, pairs)
        result = check_mirror(reference, mobile, pairs, allow_reflection=True)
        self.assertTrue(result["is_mirror_candidate"])
        self.assertAlmostEqual(result["rmsd"], 0.0, places=9)


class EnsembleTests(unittest.TestCase):
    def test_ensemble_report_is_traceable_and_inputs_remain_unchanged(self):
        reference = tetra_frame()
        conformer_b_coords = reference.coordinates + [1.0, 0.0, 0.0]
        conformer_b_coords[5] += [0.0, 0.5, 0.0]
        conformer_b = Structure(
            [Atom(i, "C", conformer_b_coords[i], serial=i + 1) for i in range(6)]
        )
        conformer_c = Structure(
            [Atom(i, "C", reference.atom(i).xyz + [0.0, 2.0, 0.0], serial=i + 1) for i in range(6)]
        )
        originals = [structure.coordinates.copy() for structure in (conformer_b, conformer_c)]
        pairs_b = match_atoms(reference, conformer_b, explicit_pairs=[(i, i) for i in range(6)])
        pairs_c = match_atoms(reference, conformer_c, explicit_pairs=[(i, i) for i in range(6)])

        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "reference.xyz"
            b_path = Path(directory) / "b.xyz"
            c_path = Path(directory) / "c.xyz"
            from molecular_toolkit import save_structure

            save_structure(reference, ref_path)
            save_structure(conformer_b, b_path)
            save_structure(conformer_c, c_path)
            loaded_reference = load_structure(ref_path)
            loaded = [Conformer.from_file(path) for path in (b_path, c_path)]
            comparisons = compare_ensemble(
                loaded_reference,
                list(zip(loaded, [pairs_b, pairs_c])),
                stable_reference_keys=[0, 1, 2, 3],
                regions={"tail": ([4, 5], [4, 5])},
            )
            report_path = Path(directory) / "report.json"
            save_comparison_report(
                comparisons,
                report_path,
                reference=loaded_reference,
                aligned_directory=Path(directory) / "aligned",
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            aligned_b = load_structure(Path(directory) / "aligned" / "b_aligned.xyz")

        self.assertEqual(report["conformer_count"], 2)
        self.assertEqual(report["conformers"][0]["conformer"]["source_path"], str(b_path))
        self.assertEqual(len(report["conformers"][0]["correspondence"]["pairs"]), 6)
        self.assertIn("tail", report["conformers"][0]["alignment"]["regions"])
        np.testing.assert_allclose(
            aligned_b.atom(4).xyz, reference.atom(4).xyz, atol=1e-7
        )
        np.testing.assert_allclose(
            aligned_b.atom(5).xyz, reference.atom(5).xyz + [0.0, 0.5, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(conformer_b.coordinates, originals[0])
        np.testing.assert_allclose(conformer_c.coordinates, originals[1])

    def test_region_motion_is_split_from_internal_deformation(self):
        coordinates = np.vstack(
            [
                tetra_frame().coordinates[:4],
                np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]),
            ]
        )
        reference = Structure([Atom(i, "C", coordinates[i], serial=i + 1) for i in range(7)])
        local_rotation = np.array(
            [
                [np.cos(0.25), -np.sin(0.25), 0.0],
                [np.sin(0.25), np.cos(0.25), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mobile_coords = reference.coordinates.copy()
        local_indices = np.array([4, 5, 6])
        local_ref = mobile_coords[local_indices]
        local_center = local_ref.mean(axis=0)
        mobile_coords[local_indices] = 1.1 * ((local_ref - local_center) @ local_rotation.T) + local_center
        mobile = Structure([Atom(i, "C", mobile_coords[i], serial=i + 1) for i in range(7)])
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(7)]).pairs

        result = align_on_stable_atoms(
            reference,
            mobile,
            pairs,
            stable_reference_keys=[0, 1, 2, 3],
            regions={"hinge": ([4, 5, 6], [4, 5, 6])},
        )
        hinge = result.region("hinge")

        self.assertEqual(hinge.rigid_fit_status, "fitted")
        self.assertAlmostEqual(hinge.local_rotation_angle, 14.3239448782706, places=6)
        radius_rmsd = np.sqrt(np.mean(np.sum((local_ref - local_center) ** 2, axis=1)))
        self.assertGreater(hinge.rmsd, hinge.internal_rmsd)
        self.assertAlmostEqual(hinge.internal_rmsd, 0.1 * radius_rmsd, places=8)

    def test_project_preserves_manual_choices_reference_regions_and_order(self):
        reference = tetra_frame()
        mobile_coords = reference.coordinates + [0.0, 0.0, 2.0]
        mobile_coords[4] += [0.0, 0.5, 0.0]
        mobile = Structure([Atom(i, "C", mobile_coords[i], serial=i + 1) for i in range(6)])
        original = mobile.coordinates.copy()
        correspondence = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(6)])

        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "reference.xyz"
            mobile_path = Path(directory) / "conformer.xyz"
            project_path = Path(directory) / "project.json"
            from molecular_toolkit import save_structure

            save_structure(reference, ref_path)
            save_structure(mobile, mobile_path)
            project = ComparisonProject(
                reference_path=str(ref_path),
                conformers=[
                    ConformerCorrespondenceSpec.from_correspondence(
                        mobile_path, correspondence, name="conformer"
                    )
                ],
                reference_regions={"core": [0, 1, 2, 3], "tail_anchor": [0, 1, 4, 5]},
                regions={"moving": ([4, 5], [4, 5])},
            )
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            runs = run_comparison_project(reopened, report_directory=Path(directory) / "reports")
            report = json.loads(
                (Path(directory) / "reports" / "reference_region_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["conformer_order"], [str(mobile_path)])
        self.assertTrue(reopened.conformers[0].confirmed)
        self.assertEqual(set(runs), {"core", "tail_anchor"})
        self.assertIn("moving", report["effects"][str(mobile_path)])
        np.testing.assert_allclose(mobile.coordinates, original)

    def test_alternative_candidate_confirmation_is_saved_in_ensemble_report(self):
        reference = tetra_frame(serials=(1, 2, 3, 4, 9, 9))
        swapped = reference.coordinates.copy()
        swapped[[4, 5]] = swapped[[5, 4]]
        mobile = Structure(
            [Atom(i, "C", swapped[i], serial=(1, 2, 3, 4, 9, 9)[i]) for i in range(6)]
        )
        candidates = candidates_from_identity(reference, mobile)
        selected = confirm_candidate(candidates[1], "Tail orientation manually accepted.")
        conformer = Conformer("sym", mobile)

        comparison = compare_conformer(
            reference,
            conformer,
            selected,
            stable_reference_keys=[0, 1, 2, 3],
            regions={"tail": ([4, 5], [4, 5])},
            alternative_candidates=candidates,
        )

        record = comparison.to_dict()
        self.assertEqual(record["correspondence_status"], "confirmed")
        self.assertEqual(len(record["alternative_candidates"]), 2)
        self.assertTrue(record["correspondence"]["confirmed"])


class RelativeMotionTests(unittest.TestCase):
    def test_relative_motion_series_reports_ranges_approach_and_local_motion(self):
        coordinates = np.vstack(
            [
                tetra_frame().coordinates[:4],
                np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
                np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]),
                np.array([[2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]),
            ]
        )
        reference = Structure([Atom(i, "C", coordinates[i], serial=i + 1) for i in range(10)])
        first = Structure([Atom(i, "C", coordinates[i], serial=i + 1) for i in range(10)])
        closer_coords = coordinates.copy()
        closer_coords[[6, 7], 0] += 1.0
        second = Structure([Atom(i, "C", closer_coords[i], serial=i + 1) for i in range(10)])
        pairs_first = match_atoms(reference, first, explicit_pairs=[(i, i) for i in range(10)])
        pairs_second = match_atoms(reference, second, explicit_pairs=[(i, i) for i in range(10)])
        selected = CorrespondenceCandidate("cand-000", pairs_first.pairs, confirmed=True)
        swapped = CorrespondenceCandidate(
            "cand-001",
            [
                (0, 0), (1, 1), (2, 2), (3, 3),
                (4, 8), (5, 9), (6, 6), (7, 7), (8, 4), (9, 5),
            ],
        )
        series = __import__(
            "molecular_toolkit", fromlist=["analyze_relative_motion_series"]
        ).analyze_relative_motion_series(
            reference,
            [Conformer("first", first), Conformer("second", second)],
            [selected, pairs_second],
            stable_reference_keys=[0, 1, 2, 3],
            region_a="a",
            region_b="b",
            region_definitions={"a": ([4, 5], None), "b": ([6, 7], None)},
            alternative_candidates_by_conformer=[[selected, swapped], None],
        )
        self.assertAlmostEqual(series.points[0].distance, np.sqrt(4.5), places=9)
        self.assertAlmostEqual(series.points[0].angle_deg, 90.0, places=9)
        self.assertEqual(series.points[1].approach, "approaching")
        self.assertAlmostEqual(series.angle_intervals()[0].lower, 45.0, places=9)
        self.assertAlmostEqual(series.angle_intervals()[0].upper, 90.0, places=9)
        self.assertIsNone(series.points[0].region_a.local_rotation_angle)
        self.assertEqual(series.points[0].region_a.status, "underdetermined_rotation")
        self.assertIn("adopted_atoms", series.trace_point(0))

    def test_relative_series_intervals_missing_atoms_and_reopen_traceability(self):
        base = tetra_frame()
        reference = Structure(
            [Atom(i, "C", base.coordinates[i], serial=i + 1) for i in range(6)]
            + [
                Atom(6, "C", [2.0, 2.0, 0.0], serial=7),
                Atom(7, "C", [3.0, 3.0, 0.0], serial=8),
            ]
        )
        coords_a = reference.coordinates.copy()
        coords_a[[4, 5]] += [2.0, 0.0, 0.0]
        conformer_a = Structure([Atom(i, "C", coords_a[i], serial=i + 1) for i in range(8)])
        coords_b = reference.coordinates.copy()
        coords_b[4] += [1.0, 0.0, 0.0]
        # Atom 5 is absent. Only the available region atom is used; the angle remains null.
        conformer_b = Structure(
            [Atom(i, "C", coords_b[i], serial=i + 1) for i in range(5)]
            + [Atom(5, "C", coords_b[6], serial=7), Atom(6, "C", coords_b[7], serial=8)]
        )
        pairs_a = match_atoms(reference, conformer_a, explicit_pairs=[(i, i) for i in range(8)])
        pairs_b = match_atoms(
            reference,
            conformer_b,
            explicit_pairs=[(i, i) for i in range(5)] + [(6, 5), (7, 6)],
        )
        alternative = CorrespondenceCandidate(
            candidate_id="alt-far",
            pairs=[
                (0, 1), (1, 0), (2, 2), (3, 5),
                (4, 6), (5, 3), (6, 4), (7, 7),
            ],
            fit_error="synthetic alternative",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_path = root / "reference.xyz"
            a_path = root / "a.xyz"
            b_path = root / "b.xyz"
            project_path = root / "project.json"
            from molecular_toolkit import save_structure

            save_structure(reference, ref_path)
            save_structure(conformer_a, a_path)
            save_structure(conformer_b, b_path)
            spec_a = ConformerCorrespondenceSpec.from_correspondence(
                a_path,
                pairs_a,
                name="a",
                candidate_id="auto",
                alternatives=[
                    CorrespondenceCandidate(
                        candidate_id="auto",
                        pairs=[(i, i) for i in range(8)],
                        confirmed=True,
                    ),
                    alternative,
                ],
            )
            spec_b = ConformerCorrespondenceSpec.from_correspondence(
                b_path, pairs_b, name="b"
            )
            project = ComparisonProject(
                reference_path=str(ref_path),
                conformers=[spec_a, spec_b],
                reference_regions={"core": [0, 1, 2, 3]},
                regions={
                    "tail_a": ([4], None),
                    "tail_b": ([5], None),
                },
            )
            decision = confirm_sequence_correspondence(
                project, 0, 0, candidate_id="auto", note="Tail mapping reviewed."
            )
            affected = decision_affected_measurements(project, decision.decision_id)
            self.assertEqual(affected, [])
            payload = run_relative_motion_project(
                project,
                "tail_motion",
                region_a="tail_a",
                region_b="tail_b",
                reference_region="core",
            )
            affected = decision_affected_measurements(project, decision.decision_id)
            self.assertEqual([item["conformer_index"] for item in affected], [0, 1])
            self.assertEqual(affected[1].get("adjacency"), "change_from_affected_previous")
            self.assertEqual(payload["points"][0]["approach"], None)
            self.assertEqual(payload["points"][1]["angle_deg"], None)
            self.assertEqual(payload["points"][1]["region_b"]["status"], "no_available_atoms")
            self.assertEqual(payload["points"][1]["region_b"]["missing_reference_keys"], [5])
            interval = payload["distance_intervals"][0]
            self.assertLessEqual(interval["lower"], interval["selected"])
            self.assertGreaterEqual(interval["upper"], interval["selected"])
            self.assertEqual(interval["status"], "ambiguous")
            self.assertIsNone(payload["angle_intervals"][1]["upper"])
            self.assertEqual(payload["points"][0]["active_decisions"][0]["decision_id"], decision.decision_id)

            withdraw_correspondence_decision(project, decision.decision_id)
            self.assertFalse(project.correspondence_decisions[0].active)
            self.assertFalse(project.conformers[0].confirmed)
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            self.assertEqual(reopened.relative_motion_analyses["tail_motion"]["points"][1]["source_path"], str(b_path))
            self.assertEqual(
                reopened.relative_motion_analyses["tail_motion"]["points"][0]["adopted_atoms"][0],
                {"reference_key": 0, "conformer_key": 0},
            )
            self.assertFalse(reopened.correspondence_decisions[0].active)


if __name__ == "__main__":
    unittest.main()
