import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ReflectionAlignmentError,
    Structure,
    align_on_stable_atoms,
    candidates_from_identity,
    check_mirror,
    compare_ensemble,
    Conformer,
    enumerate_candidates,
    load_structure,
    lock_candidate_pairs,
    match_atoms,
    save_comparison_report,
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


if __name__ == "__main__":
    unittest.main()
