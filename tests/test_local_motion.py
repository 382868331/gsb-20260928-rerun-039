import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molecular_toolkit import (
    Atom,
    AtomCorrespondence,
    ComparisonSequence,
    Conformer,
    Project,
    Structure,
    align_on_stable_atoms,
    compare_reference_frames,
    explain_frame_difference,
    match_atoms,
    save_structure,
)


TETRA = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def make_structure(coordinates):
    return Structure([Atom(i, "C", coordinates[i], serial=i + 1) for i in range(len(coordinates))])


def two_cluster_frames(local_rotation=0.0, perturbation=0.0, region_offset=(10.0, 0.0, 0.0)):
    stable = TETRA
    region = TETRA + np.asarray(region_offset)
    reference = make_structure(np.vstack([stable, region]))

    mobile_coords = reference.coordinates.copy()
    rotation = np.array(
        [
            [np.cos(local_rotation), -np.sin(local_rotation), 0.0],
            [np.sin(local_rotation), np.cos(local_rotation), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    centroid = region.mean(axis=0)
    mobile_coords[4:] = (mobile_coords[4:] - centroid) @ rotation.T + centroid
    if perturbation:
        mobile_coords[4] += [0.0, 0.0, perturbation]
    mobile = make_structure(mobile_coords)
    return reference, mobile


class LocalMotionTests(unittest.TestCase):
    def test_motion_split_into_translation_rotation_strain(self):
        reference, mobile = two_cluster_frames(local_rotation=0.5, perturbation=0.3)
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(8)]).pairs
        result = align_on_stable_atoms(
            reference,
            mobile,
            pairs,
            stable_reference_keys=[0, 1, 2, 3],
            regions={"moving": ([4, 5, 6, 7], [4, 5, 6, 7])},
        )
        motion = result.region("moving").motion
        self.assertFalse(motion.degenerate)
        self.assertAlmostEqual(motion.rotation_angle_deg, np.degrees(0.5), delta=0.5)
        self.assertLess(motion.translation_distance, 0.1)
        # Optimal local rotation absorbs part of the single-atom displacement;
        # the remainder is the internal deformation and is therefore below 0.15.
        self.assertAlmostEqual(motion.internal_rmsd, 0.1203588, places=5)
        self.assertGreater(motion.rigid_rmsd, 0.0)
        self.assertAlmostEqual(
            motion.total_rmsd**2, motion.rigid_rmsd**2 + motion.internal_rmsd**2, places=9
        )

    def test_motion_absorbed_by_global_alignment_is_still_local(self):
        reference, mobile = two_cluster_frames()
        mobile.atoms[4].xyz += [5.0, 0.0, 0.0]
        mobile.atoms[5].xyz += [5.0, 0.0, 0.0]
        mobile.atoms[6].xyz += [5.0, 0.0, 0.0]
        mobile.atoms[7].xyz += [5.0, 0.0, 0.0]
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(8)]).pairs
        result = align_on_stable_atoms(
            reference, mobile, pairs, [0, 1, 2, 3], regions={"moving": ([4, 5, 6, 7], [4, 5, 6, 7])}
        )
        motion = result.region("moving").motion
        self.assertAlmostEqual(motion.translation_distance, 5.0, places=9)
        self.assertAlmostEqual(motion.rotation_angle_deg, 0.0, places=9)
        self.assertAlmostEqual(motion.internal_rmsd, 0.0, places=9)

    def test_degenerate_region_reports_translation_only(self):
        reference, mobile = two_cluster_frames()
        mobile.atoms[4].xyz += [2.0, 0.0, 0.0]
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(8)]).pairs
        result = align_on_stable_atoms(
            reference, mobile, pairs, [0, 1, 2, 3], regions={"tip": ([4], [4])}
        )
        motion = result.region("tip").motion
        self.assertTrue(motion.degenerate)
        self.assertAlmostEqual(motion.translation_distance, 2.0, places=9)
        self.assertIsNone(motion.rotation)


class ReferenceFrameTests(unittest.TestCase):
    def test_different_frames_explain_region_differences(self):
        # Two stationary clusters and one moving region.
        stable_a = TETRA
        stable_b = TETRA + [0.0, 10.0, 0.0]
        moving = TETRA + [10.0, 0.0, 0.0]
        reference = make_structure(np.vstack([stable_a, stable_b, moving]))
        mobile_coords = reference.coordinates.copy()
        mobile_coords[8:] += [0.0, 3.0, 0.0]
        mobile_coords[8] += [0.0, 0.0, 0.2]
        mobile = make_structure(mobile_coords)
        pairs = match_atoms(reference, mobile, explicit_pairs=[(i, i) for i in range(12)]).pairs
        conformers = [(Conformer("c1", mobile), AtomCorrespondence(pairs))]

        studies = compare_reference_frames(
            reference,
            conformers,
            {"cluster_a": [0, 1, 2, 3], "cluster_b": [4, 5, 6, 7]},
            regions={"moving": ([8, 9, 10, 11], None)},
        )
        explanation = explain_frame_difference(studies, "moving")
        entries = explanation["frames"]
        self.assertEqual(set(entries), {"cluster_a", "cluster_b"})
        self.assertLess(explanation["internal_rmsd_spread"], 1e-9)
        for frame in entries.values():
            self.assertAlmostEqual(frame["translation_distance"], 3.0, places=3)
            self.assertAlmostEqual(frame["internal_rmsd"], 0.0803694, places=5)


class SequenceTests(unittest.TestCase):
    def test_sequence_preserves_conformer_order(self):
        reference, mobile_a = two_cluster_frames(region_offset=(10.0, 0.0, 0.0))
        mobile_b = make_structure(reference.coordinates.copy())
        mobile_b.atoms[4].xyz += [0.0, 0.0, 0.5]
        from molecular_toolkit import Conformer

        pair_list = match_atoms(reference, mobile_a, explicit_pairs=[(i, i) for i in range(8)]).pairs
        pairs = AtomCorrespondence(pair_list)
        sequence = ComparisonSequence(
            __import__("molecular_toolkit").compare_ensemble(
                reference,
                [
                    (Conformer("a", mobile_a), pairs),
                    (Conformer("b", mobile_b), pairs),
                    (Conformer("c", mobile_a), pairs),
                ],
                [0, 1, 2, 3],
                regions={"moving": ([4, 5, 6, 7], [4, 5, 6, 7])},
            )
        )
        self.assertEqual(sequence.names, ["a", "b", "c"])
        self.assertEqual(sequence.step(0, 1).conformer.name, "b")
        transitions = sequence.transitions()
        self.assertEqual([(item["from"], item["to"]) for item in transitions], [("a", "b"), ("b", "c")])
        self.assertNotAlmostEqual(
            transitions[0]["regions"]["moving"]["rmsd_delta"], 0.0
        )


class ProjectPersistenceTests(unittest.TestCase):
    def test_save_and_reload_reproduces_manual_choices(self):
        reference, mobile = two_cluster_frames(local_rotation=0.3)
        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "reference.xyz"
            mob_path = Path(directory) / "c1.xyz"
            save_structure(reference, ref_path)
            save_structure(mobile, mob_path)

            project = Project.from_files(ref_path, [mob_path])
            project.set_manual_pairs(0, [(i, i) for i in range(8)])
            swapped = [(i, i) for i in range(8)]
            swapped[4], swapped[5] = (4, 5), (5, 4)
            project.add_alternative_pairs(0, [swapped])
            project.lock_pairs(0, [(0, 0), (1, 1)])
            project.set_stable_region([0, 1, 2, 3])
            project.set_region("moving", [4, 5, 6, 7], [4, 5, 6, 7])
            comparisons = project.run()
            before = comparisons[0].alignment.region("moving").motion.to_dict()
            project_path = Path(directory) / "project.json"
            project.save(project_path)

            reopened = Project.load(project_path)
            self.assertEqual(reopened.manual_pairs[0], [(i, i) for i in range(8)])
            self.assertEqual(reopened.locked_pairs[0], [(0, 0), (1, 1)])
            self.assertEqual(len(reopened.pending_alternatives[0]), 1)
            self.assertEqual(reopened.stable_reference_keys, [0, 1, 2, 3])
            self.assertIn("moving", reopened.regions)
            self.assertIsNotNone(reopened.comparisons)
            after = reopened.comparisons[0].alignment.region("moving").motion.to_dict()
            self.assertAlmostEqual(
                before["rotation_angle_deg"], after["rotation_angle_deg"], places=12
            )
            self.assertAlmostEqual(before["internal_rmsd"], after["internal_rmsd"], places=12)

            reopened.confirm_alternative(0, 0)
            self.assertEqual(reopened.manual_pairs[0][4], (4, 5))

            data = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(data["conformers"][0]["manual_pairs"][4], {"reference_key": 4, "mobile_key": 4})

    def test_reload_detects_changed_source_coordinates(self):
        reference, mobile = two_cluster_frames()
        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "reference.xyz"
            mob_path = Path(directory) / "c1.xyz"
            save_structure(reference, ref_path)
            save_structure(mobile, mob_path)
            project = Project.from_files(ref_path, [mob_path])
            project.set_manual_pairs(0, [(i, i) for i in range(8)])
            project.set_stable_region([0, 1, 2, 3])
            project.run()
            project_path = Path(directory) / "project.json"
            project.save(project_path)
            mobile.atoms[0].xyz += [1.0, 0.0, 0.0]
            save_structure(mobile, mob_path)
            with self.assertRaises(ValueError):
                Project.load(project_path)


if __name__ == "__main__":
    unittest.main()
