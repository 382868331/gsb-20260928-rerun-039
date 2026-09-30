import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    Structure,
    align_structures,
    build_correspondence_candidates,
    compare_ensemble,
    lock_candidate_pairs,
    save_ensemble_report,
)


def make_structure(prefix="C", coordinates=None, serials=None):
    if coordinates is None:
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.3],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
    if serials is None:
        serials = [1, 2, 3, 10, 10, 11]
    return Structure(
        [Atom(i, prefix, coordinates[i], serial=serials[i]) for i in range(len(coordinates))],
        metadata={"name": prefix},
    )


class EnsembleTests(unittest.TestCase):
    def test_local_regions_do_not_get_masked_by_global_rmsd(self):
        reference = make_structure()
        mobile_coordinates = reference.coordinates.copy()
        mobile_coordinates[4:] += np.array([0.0, 0.0, 2.0])
        mobile = make_structure("C", mobile_coordinates)
        correspondence = lock_pairs(reference, mobile, [(i, i) for i in range(6)])

        result = align_structures(
            reference,
            mobile,
            correspondence,
            fit_reference_keys=[0, 1, 2, 3],
            regions={"stable": [0, 1, 2, 3], "loop": [4, 5]},
        )

        self.assertAlmostEqual(result.region_statistics["stable"]["rmsd"], 0.0)
        self.assertAlmostEqual(result.region_statistics["loop"]["rmsd"], 2.0)
        self.assertAlmostEqual(result.rmsd, 0.0)
        self.assertTrue(all(item["used_for_fit"] for item in result.paired_errors[:4]))
        self.assertFalse(any(item["used_for_fit"] for item in result.paired_errors[4:]))

    def test_enumerate_symmetric_candidates_and_lock_relationship(self):
        reference = make_structure()
        swapped = reference.coordinates.copy()
        swapped[3], swapped[4] = swapped[4].copy(), swapped[3].copy()
        mobile = make_structure("C", swapped, [1, 2, 3, 10, 10, 11])

        candidates = build_correspondence_candidates(
            reference,
            mobile,
            locked_pairs=[(0, 0), (1, 1), (2, 2), (5, 5)],
            ambiguous_groups={"symmetric_pair": {"reference": [3, 4], "mobile": [3, 4]}},
            identity="none",
        )

        self.assertEqual(len(candidates), 2)
        swapped_candidate = next(
            candidate
            for candidate in candidates
            if (3, 4) in candidate.correspondence.pairs
        )
        locked = lock_candidate_pairs(candidates[1], [(3, 3)], reference, mobile)
        self.assertIn((3, 3), locked.correspondence.pairs)
        self.assertIn((3, 3), locked.locked_pairs)
        self.assertEqual(len(locked.correspondence.pairs), 5)
        self.assertEqual(len({pair[0] for pair in locked.correspondence.pairs}), 5)
        self.assertEqual(len({pair[1] for pair in locked.correspondence.pairs}), 5)
        self.assertEqual(locked.correspondence.unmatched_reference, [4])
        self.assertEqual(len(locked.correspondence.unmatched_mobile), 1)

    def test_missing_atoms_and_mirror_candidate_are_reported_separately(self):
        reference = make_structure()
        reflection = np.diag([1.0, 1.0, -1.0])
        mobile = Structure(
            [Atom(i, "C", reflection @ reference.atom(i).xyz, serial=i + 1) for i in range(5)]
            + [Atom(5, "O", [3.0, 3.0, 3.0], serial=99)],
            metadata={"source_path": "mirror.xyz"},
        )
        candidates = build_correspondence_candidates(
            reference,
            mobile,
            locked_pairs=[(i, i) for i in range(5)],
            identity="none",
            include_mirror=True,
        )
        mirror_candidate = next(candidate for candidate in candidates if candidate.is_mirror)

        ensemble = compare_ensemble(
            reference,
            [("mirror", mobile, mirror_candidate)],
            stable_keys=[0, 1, 2],
            flexible_regions={"core": [3, 4, 5]},
        )
        comparison = ensemble.comparisons[0]

        self.assertTrue(comparison.is_mirror_candidate)
        self.assertTrue(comparison.result.is_reflection)
        self.assertEqual(comparison.result.correspondence.unmatched_reference, [5])
        self.assertEqual(comparison.result.correspondence.unmatched_mobile, [5])
        self.assertEqual(comparison.region_coverage["core"]["missing_reference_keys"], [5])
        self.assertEqual(comparison.region_coverage["mobile_only"]["mobile_keys"], [5])

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "ensemble.json"
            save_ensemble_report(ensemble, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        conformation = report["conformations"][0]
        self.assertEqual(conformation["conformer_id"], "mirror")
        self.assertEqual(conformation["candidate_label"], "candidate_0_mirror")
        self.assertEqual(len(conformation["locked_pairs"]), 5)
        self.assertEqual(conformation["mobile_metadata"]["source_path"], "mirror.xyz")
        self.assertEqual(len(conformation["reference_fingerprint"]), 64)


def lock_pairs(reference, mobile, pairs):
    from molecular_toolkit import match_atoms

    correspondence = match_atoms(reference, mobile, explicit_pairs=pairs, identity="none")
    return correspondence


if __name__ == "__main__":
    unittest.main()
