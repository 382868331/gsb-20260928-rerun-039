import unittest

from molecular_toolkit import (
    ContactRule,
    RelativeMotionPoint,
    RelativeMotionSeries,
    analyze_contacts,
    analyze_joint_contacts,
    compare_joint_contact_conditions,
)


def motion_point(index, region_a, region_b, distance, missing_keys=None):
    missing_atoms = {}
    atom_pairs = {region_a: [[1, 1]], region_b: [[2, 2]]}
    if missing_keys:
        missing_atoms[region_b] = list(missing_keys)
        atom_pairs[region_b] = [] if len(missing_keys) == 3 else [[2, 2]]
    return RelativeMotionPoint(
        index=index,
        conformer_name=f"frame-{index}",
        source_path=f"frame-{index}.xyz",
        region_a=region_a,
        region_b=region_b,
        reference_region="core",
        distance=distance,
        previous_distance=None,
        distance_change=None,
        motion_state=None,
        angle_deg=None,
        angle_status="measured" if distance is not None else "missing_atoms",
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
        atom_pairs=atom_pairs,
        candidate_id=None,
        correspondence_status="auto",
        available_interpretations=1,
        missing_atoms=missing_atoms,
    )


def contact_analysis(name_a, name_b, values, enter_distance=2.0, exit_distance=3.0, missing=None):
    series = RelativeMotionSeries(
        region_a=name_a,
        region_b=name_b,
        reference_region="core",
        points=[
            motion_point(index, name_a, name_b, distance, (missing or {}).get(index))
            for index, distance in enumerate(values)
        ],
    )
    return analyze_contacts(
        series,
        ContactRule(name_a, name_b, enter_distance, exit_distance, min_contact_duration=1),
    )


class JointContactTests(unittest.TestCase):
    def test_staggered_contacts_missing_frames_and_separation(self):
        first = contact_analysis("left", "right", [1, 1, 1, None, 1, 4, 1, 1])
        second = contact_analysis("top", "bottom", [4, 1, None, 1, 1, 1, None, None], missing={2: [7, 8, 9]})

        result = analyze_joint_contacts([first, second], handoff_interval=1)
        payload = result.to_dict()

        self.assertEqual(
            [point["state"] for point in payload["points"]],
            ["blocked", "joint_contact", "unknown", "unknown", "joint_contact", "blocked", "unknown", "unknown"],
        )
        self.assertEqual(payload["blocked_ranges"], [[0, 0], [5, 5]])
        self.assertEqual(len(payload["windows"]), 1)
        window = payload["windows"][0]
        self.assertEqual(window["start_index"], 1)
        self.assertEqual(window["end_index"], 4)
        self.assertEqual(window["status"], "confirmed")
        self.assertEqual(window["confirmed_support_ranges"], [[1, 1], [4, 4]])
        self.assertEqual(window["unknown_ranges"], [[2, 3]])
        self.assertEqual(
            window["actual_support_ranges"],
            {
                "left__right": [[1, 2], [4, 4]],
                "top__bottom": [[1, 1], [3, 4]],
            },
        )
        self.assertTrue(
            any(
                reason["reason"] == "confirmed_separation" and reason["pair"] == "left__right"
                for reason in window["blocking_reasons"]
            )
        )
        self.assertTrue(
            any(
                reason["reason"] == "confirmed_separation" and reason["pair"] == "top__bottom"
                for reason in window["blocking_reasons"]
            )
        )
        unknown_at_2 = payload["points"][2]["unknown_reasons"][0]
        self.assertEqual(unknown_at_2["pair"], "top__bottom")
        self.assertEqual(unknown_at_2["evidence_state"], "missing_atoms")
        self.assertEqual(unknown_at_2["missing_atoms"], {"bottom": [7, 8, 9]})
        self.assertEqual(payload["points"][3]["unknown_reasons"][0]["pair"], "left__right")

    def test_zero_handoff_does_not_stitch_discrete_observations(self):
        first = contact_analysis("left", "right", [1, 1, 1, None, 1])
        second = contact_analysis("top", "bottom", [4, 1, None, 1, 1], missing={2: [7, 8, 9]})

        result = analyze_joint_contacts([first, second], handoff_interval=0)

        self.assertEqual(
            [(window.start_index, window.end_index, window.status) for window in result.windows],
            [(1, 1, "confirmed"), (4, 4, "confirmed")],
        )

    def test_confirmed_separation_blocks_larger_handoff(self):
        first = contact_analysis("left", "right", [1, 1, 1, 1, 1, 4, 1, 1])
        second = contact_analysis("top", "bottom", [1, 1, None, None, 1, 1, None, 1])

        result = analyze_joint_contacts([first, second], handoff_interval=2)

        self.assertEqual(
            [point["state"] for point in result.to_dict()["points"]],
            ["joint_contact", "joint_contact", "unknown", "unknown", "joint_contact", "blocked", "unknown", "joint_contact"],
        )
        self.assertEqual(
            [(window.start_index, window.end_index) for window in result.windows],
            [(0, 4), (6, 7)],
        )

    def test_threshold_comparison_changes_joint_windows(self):
        def conditions(enter_distance, exit_distance):
            first = contact_analysis("left", "right", [1, 2.5, 1], enter_distance, exit_distance)
            second = contact_analysis("top", "bottom", [1, 1, 1], enter_distance, exit_distance)
            return [first, second], 0

        comparison = compare_joint_contact_conditions(
            {"strict": conditions(2.0, 2.0), "hysteresis": conditions(2.0, 3.0)}
        )

        self.assertEqual(
            [(window.start_index, window.end_index) for window in comparison["strict"].windows],
            [(0, 0), (2, 2)],
        )
        self.assertEqual(
            [(window.start_index, window.end_index) for window in comparison["hysteresis"].windows],
            [(0, 2)],
        )

    def test_result_is_identical_under_input_rigid_transform(self):
        from tests.test_contact_missing_atoms import rigid_transform, run_custom_contact_analysis

        values = [0.2, 0.45, None, 0.2, 0.9]
        keys = [[7, 8, 9], [7, 8, 9], [7, 8], [7, 8, 9], [7, 8, 9]]
        baseline, _ = run_custom_contact_analysis(values, keys)
        moved, _ = run_custom_contact_analysis(values, keys, transform=rigid_transform())

        baseline_joint = analyze_joint_contacts([baseline, baseline], handoff_interval=1)
        moved_joint = analyze_joint_contacts([moved, moved], handoff_interval=1)

        self.assertEqual(
            [point["state"] for point in moved_joint.to_dict()["points"]],
            [point["state"] for point in baseline_joint.to_dict()["points"]],
        )


if __name__ == "__main__":
    unittest.main()
