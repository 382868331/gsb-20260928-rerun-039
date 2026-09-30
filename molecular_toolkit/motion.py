from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GeometryError, kabsch_transforms


@dataclass(eq=False)
class LocalMotion:
    """Rigid/non-rigid split of one region after the global alignment.

    Coordinates are compared in the stable reference frame produced by the
    global fit. The local Kabsch fit separates the region's own translation
    and rotation from its internal deformation, so motion absorbed by the
    global alignment is never reported as "no local change": the translation
    is measured between centroids in the stable frame.
    """

    atom_count: int
    reference_centroid: np.ndarray
    mobile_centroid: np.ndarray
    translation: np.ndarray
    translation_distance: float
    rotation: np.ndarray | None
    rotation_axis: np.ndarray | None
    rotation_angle_deg: float | None
    internal_rmsd: float | None
    rigid_rmsd: float | None
    total_rmsd: float
    degenerate: bool = False
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "atom_count": self.atom_count,
            "reference_centroid": self.reference_centroid.tolist(),
            "mobile_centroid": self.mobile_centroid.tolist(),
            "translation": self.translation.tolist(),
            "translation_distance": self.translation_distance,
            "rotation": None if self.rotation is None else self.rotation.tolist(),
            "rotation_axis": None if self.rotation_axis is None else self.rotation_axis.tolist(),
            "rotation_angle_deg": self.rotation_angle_deg,
            "internal_rmsd": self.internal_rmsd,
            "rigid_rmsd": self.rigid_rmsd,
            "total_rmsd": self.total_rmsd,
            "degenerate": self.degenerate,
            "note": self.note,
        }


def decompose_local_motion(
    reference_points: np.ndarray,
    aligned_mobile_points: np.ndarray,
) -> LocalMotion:
    """Split post-alignment point motion into translation, rotation and strain.

    ``aligned_mobile_points`` must already carry the global transform fitted on
    the stable region. Returns centroid translation in that stable frame, the
    region-local proper rotation and the internal-deformation RMSD left after
    the local rigid fit.
    """
    reference_points = np.asarray(reference_points, dtype=float)
    aligned_mobile_points = np.asarray(aligned_mobile_points, dtype=float)
    if reference_points.shape != aligned_mobile_points.shape or reference_points.ndim != 2:
        raise ValueError("Local motion needs two matching Nx3 point arrays")
    if len(reference_points) == 0:
        raise ValueError("Local motion needs at least one paired atom")

    reference_centroid = reference_points.mean(axis=0)
    mobile_centroid = aligned_mobile_points.mean(axis=0)
    translation = mobile_centroid - reference_centroid
    differences = aligned_mobile_points - reference_points
    total_msd = float(np.mean(np.sum(differences**2, axis=1)))
    total_rmsd = float(np.sqrt(total_msd))

    rotation: np.ndarray | None = None
    rotation_axis: np.ndarray | None = None
    rotation_angle: float | None = None
    internal_rmsd: float | None = None
    rigid_msd = 0.0
    degenerate = False
    note: str | None = None

    if len(reference_points) < 3:
        degenerate = True
        note = "Fewer than three paired atoms: only translation is defined."
    else:
        try:
            local_rotation, _, _, local_rmsd, _ = kabsch_transforms(
                reference_points, aligned_mobile_points
            )
            rotation = local_rotation
            rotation_axis, rotation_angle = _axis_angle(rotation)
            internal_msd = float(local_rmsd**2)
            internal_rmsd = float(local_rmsd)
            rigid_msd = float(np.sqrt(max(0.0, total_msd - internal_msd)))
        except (GeometryError, np.linalg.LinAlgError) as exc:
            degenerate = True
            note = f"Local rotation is underdetermined: {exc}"

    result_motion = LocalMotion(
        atom_count=len(reference_points),
        reference_centroid=reference_centroid,
        mobile_centroid=mobile_centroid,
        translation=translation,
        translation_distance=float(np.linalg.norm(translation)),
        rotation=rotation,
        rotation_axis=rotation_axis,
        rotation_angle_deg=rotation_angle,
        internal_rmsd=internal_rmsd,
        rigid_rmsd=rigid_msd,
        total_rmsd=total_rmsd,
        degenerate=degenerate,
        note=note,
    )
    return result_motion


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    cos_angle = (float(np.trace(rotation)) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = float(np.degrees(np.arccos(cos_angle)))
    raw_axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    norm = float(np.linalg.norm(raw_axis))
    if norm <= 1e-12:
        return np.zeros(3), 0.0
    return raw_axis / norm, angle
