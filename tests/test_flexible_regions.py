import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ContactRule,
    ComparisonProject,
    ConformerCorrespondenceSpec,
    RelativeMotionSeries,
    RelativeMotionPoint,
    analyze_contacts,
    analyze_relative_motion,
    confirm_project_correspondence,
    ReflectionAlignmentError,
    Structure,
    align_on_stable_atoms,
    candidates_from_identity,
    check_mirror,
    compare_ensemble,
    compare_conformer,
    Conformer,
    confirm_candidate,
    enumerate_candidates,
    load_structure,
    load_comparison_project,
    lock_candidate_pairs,
    match_atoms,
    run_comparison_project,
    save_structure,
    withdraw_project_correspondence,
    save_comparison_report,
    save_comparison_project,
    preview_project_correspondence_contact_impact,
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

    def test_relative_motion_keeps_missing_angle_as_null_and_records_atoms(self):
        coordinates = np.vstack(
            [
                tetra_frame().coordinates[:4],
                np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]]),
                np.array([[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 0.0, 1.0]]),
            ]
        )
        reference = Structure([Atom(i, "C", coordinates[i], serial=i + 1) for i in range(10)])
        first_coords = np.delete(reference.coordinates, 4, axis=0)
        first = Structure([Atom(i, "C", first_coords[i], serial=i + 1) for i in range(9)])
        second_coords = reference.coordinates.copy()
        second_coords[6:9, 0] -= 0.5
        second = Structure([Atom(i, "C", second_coords[i], serial=i + 1) for i in range(10)])
        pairs_first = match_atoms(
            reference, first, explicit_pairs=[(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6), (8, 7)]
        )
        pairs_second = match_atoms(reference, second, explicit_pairs=[(i, i) for i in range(10)])
        conformers = [Conformer("first", first), Conformer("second", second)]
        comparisons = compare_ensemble(
            reference,
            list(zip(conformers, [pairs_first, pairs_second])),
            stable_reference_keys=[0, 1, 2, 3],
            regions={"left": ([4, 5], None), "right": ([6, 7, 8], None)},
        )

        series = analyze_relative_motion(comparisons, "left", "right", reference_region="core")

        self.assertIsInstance(series, RelativeMotionSeries)
        self.assertIsNone(series.points[0].angle_deg)
        self.assertEqual(series.points[0].angle_status, "missing_atoms")
        self.assertIsNone(series.points[0].distance)
        self.assertEqual(series.points[0].missing_atoms, {"left": [4]})
        self.assertEqual(series.points[0].atom_pairs["left"], [[5, 4]])
        self.assertIsNone(series.points[1].motion_state)
        self.assertIsNotNone(series.points[1].angle_deg)

    def test_project_saves_relative_motion_and_limited_correspondence_withdrawal(self):
        reference_coords = np.vstack(
            [
                tetra_frame().coordinates[:4],
                np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
                np.array([[4.0, 0.0, 0.0], [4.0, 1.0, 0.0]]),
            ]
        )
        reference = Structure(
            [Atom(i, "C", reference_coords[i], serial=(1, 2, 3, 4, 9, 9, 7, 8)[i]) for i in range(8)]
        )
        swapped = reference.coordinates.copy()
        swapped[[4, 5]] = swapped[[5, 4]]
        ambiguous = Structure(
            [Atom(i, "C", swapped[i], serial=(1, 2, 3, 4, 9, 9, 7, 8)[i]) for i in range(8)]
        )
        clear_mobile = Structure(
            [Atom(i, "C", reference.atom(i).xyz + [0.0, 0.0, 0.2], serial=(1, 2, 3, 4, 9, 9, 7, 8)[i])
             for i in range(8)]
        )
        candidates = candidates_from_identity(reference, ambiguous)
        clear_pairs = match_atoms(reference, clear_mobile, explicit_pairs=[(i, i) for i in range(8)])

        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "reference.xyz"
            amb_path = Path(directory) / "ambiguous.xyz"
            clear_path = Path(directory) / "clear.xyz"
            project_path = Path(directory) / "project.json"
            save_structure(reference, ref_path)
            save_structure(ambiguous, amb_path)
            save_structure(clear_mobile, clear_path)
            project = ComparisonProject(
                reference_path=str(ref_path),
                conformers=[
                    ConformerCorrespondenceSpec.from_correspondence(
                        amb_path,
                        candidates[1].to_correspondence(),
                        name="ambiguous",
                        alternatives=candidates,
                    ),
                    ConformerCorrespondenceSpec.from_correspondence(
                        clear_path, clear_pairs, name="clear"
                    ),
                ],
                reference_regions={"core": [0, 1, 2, 3]},
                regions={"left": ([4, 5], None), "right": ([6, 7], None)},
                relative_motion_pairs=[["left", "right"]],
                contact_rules=[
                    {
                        "region_a": "left",
                        "region_b": "right",
                        "enter_distance": 3.0,
                        "exit_distance": 4.0,
                        "min_contact_duration": 1,
                    }
                ],
            )

            preview = preview_project_correspondence_contact_impact(
                project, 0, 0, candidate_id=candidates[0].candidate_id
            )
            self.assertEqual(len(preview["previews"]), 1)
            self.assertIn("before", preview["previews"][0])
            self.assertIn("after", preview["previews"][0])
            decision = confirm_project_correspondence(project, 0, 0, candidate_id=candidates[0].candidate_id)
            self.assertEqual(decision["affected_point_indices"], [0])
            self.assertEqual(decision["affected_segment_indices"], [0])
            save_comparison_project(project, project_path)
            reopened = load_comparison_project(project_path)
            run_comparison_project(reopened, report_directory=Path(directory) / "reports")
            motion = json.loads(
                (
                    Path(directory)
                    / "reports"
                    / "core"
                    / "relative_motion"
                    / "left__right.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(motion["points"][0]["candidate_id"], candidates[0].candidate_id)
            self.assertEqual(motion["points"][0]["decision_id"], "decision-001")
            self.assertEqual(motion["points"][0]["correspondence_status"], "confirmed")
            self.assertEqual(motion["points"][0]["atom_pairs"]["left"], [[4, 5], [5, 4]])
            contacts = json.loads(
                (
                    Path(directory)
                    / "reports"
                    / "core"
                    / "contacts"
                    / "left__right__3__4__min1.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(contacts["format"], "molecular-toolkit-contact-analysis")
            self.assertEqual(contacts["points"][0]["atom_pairs"]["left"], [[4, 5], [5, 4]])
            self.assertEqual(contacts["points"][0]["decision_id"], "decision-001")

            withdraw_project_correspondence(reopened, "decision-001")
            self.assertFalse(reopened.conformers[0].confirmed)
            self.assertIsNone(reopened.conformers[0].candidate_id)
            self.assertTrue(reopened.conformers[1].confirmed)
            self.assertEqual(reopened.correspondence_decisions[-1]["status"], "withdrawn")

    def test_contact_hysteresis_minimum_duration_and_missing_gap_segments(self):
        series = self._contact_series(
            [
                (1.0, None, None),
                (None, None, None),
                (1.0, None, None),
                (4.0, None, None),
            ]
        )
        analysis = analyze_contacts(
            series,
            ContactRule("left", "right", enter_distance=2.0, exit_distance=3.0, min_contact_duration=1),
        )

        self.assertEqual([event.event_type for event in analysis.events], ["formed", "lost"])
        self.assertEqual(len(analysis.intervals), 1)
        interval = analysis.intervals[0].to_dict()
        self.assertEqual(interval["start_index"], 0)
        self.assertEqual(interval["end_index"], 3)
        self.assertEqual(interval["observed_contact_duration"], 2)
        self.assertEqual(interval["unresolved_gap_duration"], 1)
        self.assertTrue(interval["has_unresolved_gap"])
        self.assertFalse(interval["continuous_observed_contact"])
        self.assertEqual(analysis.points[1].state, "unresolved")
        self.assertEqual(analysis.points[1].explanation.startswith("Required atoms"), True)
        self.assertEqual(analysis.points[0].atom_pairs["left"], [[4, 4]])

    def test_brief_near_threshold_visit_does_not_emit_event(self):
        series = self._contact_series(
            [
                (4.0, None, None),
                (1.0, None, None),
                (4.0, None, None),
                (1.0, None, None),
                (2.5, None, None),
                (1.0, None, None),
            ]
        )
        single_threshold = analyze_contacts(
            series,
            ContactRule("left", "right", enter_distance=2.0, exit_distance=2.0, min_contact_duration=2),
        )
        hysteresis = analyze_contacts(
            series,
            ContactRule("left", "right", enter_distance=2.0, exit_distance=3.0, min_contact_duration=2),
        )

        self.assertEqual(single_threshold.events, [])
        self.assertEqual(hysteresis.points[4].state, "contact_held")
        self.assertEqual([event.event_type for event in hysteresis.events], ["formed"])
        self.assertEqual(hysteresis.intervals[0].observed_contact_indices, [3, 5])
        self.assertEqual(hysteresis.intervals[0].hysteresis_hold_indices, [4])
        self.assertEqual(hysteresis.intervals[0].to_dict()["sustained_duration"], 3)

    def test_disputed_contact_evidence_keeps_all_interpretations(self):
        series = self._contact_series(
            [
                (1.0, (1.0, 4.0), "cand-auto"),
                (1.0, None, "decision-001"),
                (4.0, None, "decision-001"),
            ]
        )
        analysis = analyze_contacts(
            series,
            ContactRule("left", "right", enter_distance=2.0, exit_distance=3.0),
        )

        self.assertEqual(analysis.points[0].state, "unresolved")
        self.assertEqual(analysis.points[0].evidence_state, "interpretation_disputed")
        self.assertEqual(analysis.points[1].evidence_state, "confirmed_contact")
        self.assertEqual(len(analysis.points[0].alternative_measurements), 2)
        self.assertEqual(analysis.intervals[0].confirmed_contact_indices, [1])
        self.assertEqual(analysis.intervals[0].observed_contact_indices, [1])

    @staticmethod
    def _contact_series(values):
        points = []
        for index, (distance, distance_range, candidate_id) in enumerate(values):
            points.append(
                RelativeMotionPoint(
                    index=index,
                    conformer_name=f"frame-{index}",
                    source_path=f"frame-{index}.xyz",
                    region_a="left",
                    region_b="right",
                    reference_region="core",
                    distance=distance,
                    previous_distance=None,
                    distance_change=None,
                    motion_state=None,
                    angle_deg=None,
                    angle_status="measured" if distance is not None else "missing_atoms",
                    distance_range=list(distance_range) if distance_range is not None else None,
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
                    atom_pairs={"left": [[4, 4]], "right": [[6, 6]]},
                    candidate_id=candidate_id,
                    correspondence_status="confirmed" if candidate_id else "auto",
                    available_interpretations=2 if distance_range is not None else 1,
                    decision_id=candidate_id if str(candidate_id or "").startswith("decision-") else None,
                    alternative_measurements=[
                        {
                            "candidate_id": candidate_id,
                            "correspondence_status": "confirmed" if candidate_id else "auto",
                            "distance": distance,
                            "atom_pairs": {"left": [[4, 4]], "right": [[6, 6]]},
                        },
                        {
                            "candidate_id": "cand-other",
                            "correspondence_status": "alternative",
                            "distance": None if distance_range is None else distance_range[1],
                            "atom_pairs": {"left": [[4, 5]], "right": [[6, 6]]},
                        },
                    ],
                )
            )
        return RelativeMotionSeries("left", "right", "core", points)


if __name__ == "__main__":
    unittest.main()
