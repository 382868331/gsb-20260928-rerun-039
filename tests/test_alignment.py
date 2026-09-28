import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    ReflectionAlignmentError,
    Structure,
    align_structures,
    load_structure,
    match_atoms,
    save_alignment,
)


def make_reference(serials=(1, 2, 3, 4, 5)):
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.1, -0.2, 1.1],
            [0.7, 0.6, 0.4],
        ]
    )
    return Structure([Atom(i, "C", coordinates[i], serial=serials[i]) for i in range(5)])


class AlignmentTests(unittest.TestCase):
    def test_known_rotation_translation_with_missing_atoms(self):
        reference = make_reference()
        angle = np.deg2rad(37.0)
        expected_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_translation = np.array([2.5, -1.25, 4.0])
        matched_mobile_coordinates = (reference.coordinates[:4] @ expected_rotation.T) + expected_translation
        extra = matched_mobile_coordinates[-1] + [3.0, 0.0, 0.0]
        mobile = Structure(
            [Atom(i, "C", matched_mobile_coordinates[i], serial=i + 1) for i in range(4)]
            + [Atom(4, "O", extra, serial=99)]
        )
        original_mobile = mobile.coordinates.copy()

        correspondence = match_atoms(reference, mobile, identity="serial")
        self.assertEqual(correspondence.pairs, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(correspondence.unmatched_reference, [4])
        self.assertEqual(correspondence.unmatched_mobile, [4])

        result = align_structures(reference, mobile, correspondence)
        np.testing.assert_allclose(result.rotation, expected_rotation.T, atol=1e-10)
        np.testing.assert_allclose(result.translation, -expected_rotation.T @ expected_translation, atol=1e-10)
        self.assertAlmostEqual(result.rmsd, 0.0, places=10)
        np.testing.assert_allclose(mobile.coordinates, original_mobile)

    def test_duplicate_serials_are_not_order_matched_but_explicit_pairs_work(self):
        reference = make_reference(serials=(1, 1, 2, 3, 4))
        angle = np.deg2rad(-21.0)
        expected_rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        expected_translation = np.array([-0.5, 3.0, 1.5])
        transformed = (reference.coordinates @ expected_rotation.T) + expected_translation
        mobile = Structure(
            [Atom(i, "C", transformed[i], serial=serial) for i, serial in enumerate((1, 1, 2, 3, 4))]
        )

        automatic = match_atoms(reference, mobile, identity="serial")
        self.assertEqual(automatic.pairs, [(2, 2), (3, 3), (4, 4)])
        self.assertEqual(automatic.unmatched_reference, [0, 1])
        self.assertEqual(automatic.unmatched_mobile, [0, 1])
        self.assertIn("1", automatic.ambiguous_identities)

        explicit = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(5)])
        result = align_structures(reference, mobile, explicit)
        np.testing.assert_allclose(result.rotation, expected_rotation.T, atol=1e-10)
        np.testing.assert_allclose(result.translation, -expected_rotation.T @ expected_translation, atol=1e-10)
        self.assertTrue(explicit.is_complete)

    def test_mirror_is_not_accepted_as_ordinary_rotation(self):
        reference = make_reference()
        reflection = np.diag([1.0, 1.0, -1.0])
        mobile = Structure(
            [Atom(i, "C", reflection @ reference.atom(i).xyz, serial=i + 1) for i in range(5)]
        )
        correspondence = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(5)])

        with self.assertRaises(ReflectionAlignmentError):
            align_structures(reference, mobile, correspondence)

        improper = align_structures(reference, mobile, correspondence, allow_reflection=True)
        self.assertTrue(improper.is_reflection)
        self.assertAlmostEqual(np.linalg.det(improper.rotation), -1.0, places=10)
        self.assertAlmostEqual(improper.rmsd, 0.0, places=10)

    def test_save_aligned_structure_and_error_report(self):
        reference = make_reference()
        rotation = np.eye(3)
        translation = np.array([1.0, 2.0, 3.0])
        mobile = Structure(
            [Atom(i, "C", reference.atom(i).xyz + translation, serial=i + 1) for i in range(5)]
        )
        correspondence = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(5)])
        result = align_structures(reference, mobile, correspondence)

        with tempfile.TemporaryDirectory() as directory:
            aligned_path = Path(directory) / "aligned.xyz"
            report_path = Path(directory) / "report.json"
            save_alignment(result, report_path, aligned_path)
            loaded_aligned = load_structure(aligned_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(loaded_aligned.coordinates, reference.coordinates, atol=1e-8)
        self.assertEqual(len(report["paired_errors"]), 5)
        self.assertFalse(report["is_reflection"])


if __name__ == "__main__":
    unittest.main()
