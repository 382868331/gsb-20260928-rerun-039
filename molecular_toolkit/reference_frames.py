from __future__ import annotations

from dataclasses import dataclass

from .ensemble import Conformer, ConformerComparison, compare_ensemble
from .structures import Structure


@dataclass(eq=False)
class ReferenceFrame:
    name: str
    stable_reference_keys: list[int]


@dataclass(eq=False)
class FrameStudy:
    frame: ReferenceFrame
    comparisons: list[ConformerComparison]

    def to_dict(self) -> dict:
        return {
            "frame": self.frame.name,
            "stable_reference_keys": self.frame.stable_reference_keys,
            "conformers": [comparison.to_dict() for comparison in self.comparisons],
        }


def compare_reference_frames(
    reference: Structure | "Conformer",
    conformers: list[tuple[Conformer, object]],
    frames: dict[str, list[int]] | list[ReferenceFrame],
    regions: dict | None = None,
) -> dict[str, FrameStudy]:
    """Rerun the same ordered comparison under each candidate stable region.

    Local motion is reference-frame dependent: each named frame produces its
    own global transform and its own per-region translation/rotation/strain.
    Keeping the separate studies makes that dependency explicit instead of
    presenting one arbitrary choice as the only result.
    """
    if isinstance(reference, Conformer):
        reference = reference.structure
    normalized_frames = _normalize_frames(frames)
    studies: dict[str, FrameStudy] = {}
    for frame in normalized_frames:
        comparisons = compare_ensemble(
            reference,
            conformers,
            stable_reference_keys=frame.stable_reference_keys,
            regions=regions,
        )
        studies[frame.name] = FrameStudy(frame=frame, comparisons=comparisons)
    return studies


def explain_frame_difference(
    studies: dict[str, FrameStudy],
    region_name: str,
    conformer_index: int = 0,
) -> dict:
    """Explain how one measured region differs between reference frames.

    Reports each frame's centroid translation, local rotation and internal
    deformation, plus the spread across frames. Internal deformation is
    invariant under the global rigid alignment (all frames fit rigid
    transforms), so it should agree up to relabelling/ordering differences;
    translation and rotation differ because each frame defines the fixed
    coordinate system.
    """
    entries = {}
    for frame_name, study in studies.items():
        alignment = study.comparisons[conformer_index].alignment
        region = alignment.regions[region_name]
        motion = region.motion
        entries[frame_name] = {
            "translation_distance": None if motion is None else motion.translation_distance,
            "rotation_angle_deg": None if motion is None else motion.rotation_angle_deg,
            "internal_rmsd": None if motion is None else motion.internal_rmsd,
            "rigid_rmsd": None if motion is None else motion.rigid_rmsd,
            "total_rmsd": None if motion is None else motion.total_rmsd,
        }

    def spread(key: str) -> float:
        values = [entry[key] for entry in entries.values() if entry[key] is not None]
        if not values:
            return 0.0
        return max(values) - min(values)

    return {
        "region": region_name,
        "conformer_index": conformer_index,
        "frames": entries,
        "translation_distance_spread": spread("translation_distance"),
        "rotation_angle_spread_deg": spread("rotation_angle_deg"),
        "internal_rmsd_spread": spread("internal_rmsd"),
        "explanation": (
            "Internal deformation is independent of the chosen stable frame; "
            "translation and rotation of the region are reported in each frame's "
            "coordinate system, so their values legitimately differ across frames."
        ),
    }


def frame_summary(studies: dict[str, FrameStudy], conformer_index: int = 0) -> list[dict]:
    rows = []
    for frame_name, study in studies.items():
        alignment = study.comparisons[conformer_index].alignment
        rows.append(
            {
                "frame": frame_name,
                "stable_rmsd": alignment.stable_rmsd,
                "is_mirror_candidate": alignment.is_mirror_candidate,
                "regions": {
                    name: {
                        "rmsd": region.rmsd,
                        "translation_distance": (
                            None if region.motion is None else region.motion.translation_distance
                        ),
                        "rotation_angle_deg": (
                            None if region.motion is None else region.motion.rotation_angle_deg
                        ),
                        "internal_rmsd": (
                            None if region.motion is None else region.motion.internal_rmsd
                        ),
                    }
                    for name, region in alignment.regions.items()
                },
            }
        )
    return rows


def _normalize_frames(frames) -> list[ReferenceFrame]:
    if isinstance(frames, dict):
        return [ReferenceFrame(name=name, stable_reference_keys=list(keys)) for name, keys in frames.items()]
    normalized = []
    for frame in frames:
        if isinstance(frame, ReferenceFrame):
            normalized.append(frame)
        else:
            raise ValueError("Frames must be ReferenceFrame instances or {name: reference_keys}")
    names = [frame.name for frame in normalized]
    if len(names) != len(set(names)):
        raise ValueError("Reference frame names must be unique")
    return normalized
