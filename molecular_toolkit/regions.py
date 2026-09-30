from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .correspondence import Pair
from .geometry import ReflectionAlignmentError, kabsch_transforms
from .structures import Structure, transform_structure


AtomKeys = Iterable[int]


@dataclass(eq=False)
class RegionDisplacements:
    name: str
    reference_keys: list[int]
    mobile_keys: list[int]
    distances: list[float] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    local_translation: list[float] | None = None
    local_rotation: list[list[float]] | None = None
    local_rotation_angle: float | None = None
    local_rotation_axis: list[float] | None = None
    internal_distances: list[float] = field(default_factory=list)
    internal_vectors: list[list[float]] = field(default_factory=list)
    internal_rmsd_value: float | None = None
    rigid_fit_status: str = "not_fitted"

    @property
    def rmsd(self) -> float:
        if not self.distances:
            raise ValueError(f"Region {self.name!r} has no paired atoms")
        return float(np.sqrt(np.mean(np.square(self.distances))))

    @property
    def mean_abs_displacement(self) -> float:
        return float(np.mean(self.distances)) if self.distances else float("nan")

    @property
    def max_displacement(self) -> float:
        return float(np.max(self.distances)) if self.distances else float("nan")

    def percentile(self, value: float) -> float:
        if not self.distances:
            raise ValueError(f"Region {self.name!r} has no paired atoms")
        return float(np.percentile(self.distances, value))

    @property
    def translation_distance(self) -> float | None:
        if self.local_translation is None:
            return None
        return float(np.linalg.norm(self.local_translation))

    @property
    def internal_rmsd(self) -> float | None:
        return self.internal_rmsd_value

    def to_dict(self) -> dict:
        per_atom = [
            {
                "reference_key": ref_key,
                "mobile_key": mobile_key,
                "distance": float(distance),
                "displacement_vector": vector,
            }
            for ref_key, mobile_key, distance, vector in zip(
                self.reference_keys, self.mobile_keys, self.distances, self.vectors
            )
        ]
        return {
            "name": self.name,
            "atom_count": len(self.distances),
            "rmsd": self.rmsd if self.distances else None,
            "mean_abs_displacement": self.mean_abs_displacement if self.distances else None,
            "median_displacement": self.percentile(50) if self.distances else None,
            "max_displacement": self.max_displacement if self.distances else None,
            "p95_displacement": self.percentile(95) if self.distances else None,
            "translation_vector": self.local_translation,
            "translation_distance": self.translation_distance,
            "rotation": self.local_rotation,
            "rotation_angle_deg": self.local_rotation_angle,
            "rotation_axis": self.local_rotation_axis,
            "internal_rmsd": self.internal_rmsd_value,
            "internal_distances": self.internal_distances,
            "internal_vectors": self.internal_vectors,
            "rigid_fit_status": self.rigid_fit_status,
            "per_atom": per_atom,
        }


@dataclass(eq=False)
class RegionAlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    aligned_mobile: Structure
    stable_pairs: list[Pair]
    stable_rmsd_proper: float
    stable_rmsd_reflected: float
    is_mirror_candidate: bool
    regions: dict[str, RegionDisplacements] = field(default_factory=dict)
    stable_reference_keys: list[int] = field(default_factory=list)
    region_definitions: dict[str, dict[str, list[int] | None]] = field(default_factory=dict)

    @property
    def stable_rmsd(self) -> float:
        return self.stable_rmsd_proper

    def region(self, name: str) -> RegionDisplacements:
        return self.regions[name]

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "stable_pairs": [
                {"reference_key": ref_key, "mobile_key": mobile_key}
                for ref_key, mobile_key in self.stable_pairs
            ],
            "stable_reference_keys": self.stable_reference_keys,
            "region_definitions": self.region_definitions,
            "stable_rmsd": self.stable_rmsd_proper,
            "stable_rmsd_reflected": self.stable_rmsd_reflected,
            "is_mirror_candidate": self.is_mirror_candidate,
            "regions": {name: region.to_dict() for name, region in self.regions.items()},
        }


def align_on_stable_atoms(
    reference: Structure,
    mobile: Structure,
    pairs: Sequence[Pair],
    stable_reference_keys: AtomKeys,
    regions: dict[str, tuple[AtomKeys, AtomKeys]] | None = None,
    *,
    flag_mirror_tolerance: float = 1e-6,
    mirror_gap_tolerance: float = 1e-6,
) -> RegionAlignmentResult:
    """Fit a rigid transform on stable atoms only and report local regions.

    ``stable_reference_keys`` selects reference atom keys. Their mobile partners
    are taken from ``pairs``. ``regions`` maps a name to
    ``(reference_keys, mobile_keys)``; when mobile keys are omitted/``None``
    the pair mapping is reused. Every paired atom outside the stable set is
    automatically collected into an ``unselected`` region so a flexible local
    change cannot be hidden by the overall fit.
    """
    stable_keys = _validate_keys(stable_reference_keys, reference, "stable reference")
    stable_set = set(stable_keys)
    region_definitions: dict[str, dict[str, list[int] | None]] = {}
    pair_map = dict(pairs)
    reverse_map = {mobile_key: ref_key for ref_key, mobile_key in pairs}
    if len(reverse_map) != len(pairs):
        raise ValueError("Each mobile atom can be paired with at most one reference atom")
    missing = sorted(stable_set - set(pair_map))
    if missing:
        raise ValueError(f"Stable reference atoms lack a correspondence pair: {missing}")
    stable_pairs = sorted((ref_key, pair_map[ref_key]) for ref_key in stable_set)
    if len(stable_pairs) < 3:
        raise ValueError("At least three stable atom pairs are required for rigid fitting")

    reference_points = np.vstack([reference.atom(ref_key).xyz for ref_key, _ in stable_pairs])
    mobile_points = np.vstack([mobile.atom(mobile_key).xyz for _, mobile_key in stable_pairs])
    rotation, translation, _, rmsd_proper, rmsd_reflected = kabsch_transforms(
        reference_points, mobile_points
    )
    aligned_mobile = transform_structure(mobile, rotation, translation)
    is_mirror = (
        rmsd_reflected + flag_mirror_tolerance < rmsd_proper
        and abs(rmsd_proper - rmsd_reflected) > mirror_gap_tolerance
    )

    result = RegionAlignmentResult(
        rotation=rotation,
        translation=translation,
        aligned_mobile=aligned_mobile,
        stable_pairs=stable_pairs,
        stable_rmsd_proper=float(rmsd_proper),
        stable_rmsd_reflected=float(rmsd_reflected),
        is_mirror_candidate=bool(is_mirror),
        stable_reference_keys=stable_keys,
        region_definitions=region_definitions,
    )

    assigned_pairs: set[Pair] = set(stable_pairs)
    if regions:
        for name, value in regions.items():
            ref_keys, mobile_keys = _normalize_region(value)
            region_definitions[name] = {
                "reference_keys": ref_keys,
                "mobile_keys": mobile_keys,
            }
            region_pairs = _resolve_region_pairs(
                name, ref_keys, mobile_keys, reference, mobile, pair_map
            )
            assigned_pairs.update(region_pairs)
            result.regions[name] = _measure_region(name, reference, aligned_mobile, region_pairs)

    remaining = [pair for pair in pairs if pair not in assigned_pairs]
    if remaining:
        result.regions["unselected"] = _measure_region(
            "unselected", reference, aligned_mobile, remaining
        )
    return result


def check_mirror(
    reference: Structure,
    mobile: Structure,
    pairs: Sequence[Pair],
    *,
    allow_reflection: bool = False,
) -> dict:
    """Compare proper and improper fits for a complete candidate mapping."""
    if len(pairs) < 3:
        raise ValueError("At least three pairs are required for mirror checking")
    ref_points = np.vstack([reference.atom(ref_key).xyz for ref_key, _ in pairs])
    mob_points = np.vstack([mobile.atom(mobile_key).xyz for _, mobile_key in pairs])
    proper_r, proper_t, _, rmsd_proper, rmsd_reflected = kabsch_transforms(ref_points, mob_points)
    reflected_r, reflected_t, _, _, _ = kabsch_transforms(ref_points, mob_points, force_reflection=True)
    mirror_is_better = rmsd_reflected < rmsd_proper
    if mirror_is_better and not allow_reflection:
        raise ReflectionAlignmentError(
            "This correspondence fits better with a reflection; review symmetry/enantiomer "
            "candidates or pass allow_reflection=True."
        )
    chosen_rotation, chosen_translation, chosen_rmsd = (
        (reflected_r, reflected_t, rmsd_reflected)
        if mirror_is_better
        else (proper_r, proper_t, rmsd_proper)
    )
    return {
        "proper_rmsd": float(rmsd_proper),
        "reflected_rmsd": float(rmsd_reflected),
        "is_mirror_candidate": bool(mirror_is_better),
        "rotation": chosen_rotation,
        "translation": chosen_translation,
        "rmsd": float(chosen_rmsd),
        "is_reflection": bool(mirror_is_better),
    }


def _normalize_region(value: tuple[AtomKeys, AtomKeys | None]):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Each region must be (reference_keys, mobile_keys)")
    ref_keys, mobile_keys = value
    return list(ref_keys), None if mobile_keys is None else list(mobile_keys)


def _resolve_region_pairs(name, ref_keys, mobile_keys, reference, mobile, pair_map):
    ref_keys = _validate_keys(ref_keys, reference, f"region {name!r} reference")
    if mobile_keys is None:
        missing = [key for key in ref_keys if key not in pair_map]
        if missing:
            raise ValueError(f"Region {name!r} atoms lack correspondence: {missing}")
        return [(key, pair_map[key]) for key in ref_keys]
    mobile_keys = _validate_keys(mobile_keys, mobile, f"region {name!r} mobile")
    if len(ref_keys) != len(mobile_keys):
        raise ValueError(f"Region {name!r} reference and mobile atom counts differ")
    pairs = list(zip(ref_keys, mobile_keys))
    for ref_key, mobile_key in pairs:
        if pair_map.get(ref_key, mobile_key) != mobile_key:
            raise ValueError(
                f"Region {name!r} pair ({ref_key}, {mobile_key}) conflicts with correspondence"
            )
    return pairs


def _validate_keys(keys, structure, label):
    keys = list(keys)
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate atom keys in {label}")
    for key in keys:
        structure.atom(key)
    return keys


def _measure_region(name, reference, aligned_mobile, pairs):
    distances = []
    vectors = []
    for ref_key, mobile_key in pairs:
        vector = aligned_mobile.atom(mobile_key).xyz - reference.atom(ref_key).xyz
        vectors.append(vector.tolist())
        distances.append(float(np.linalg.norm(vector)))
    result = RegionDisplacements(
        name=name,
        reference_keys=[pair[0] for pair in pairs],
        mobile_keys=[pair[1] for pair in pairs],
        distances=distances,
        vectors=vectors,
    )
    reference_points = np.vstack([reference.atom(key).xyz for key in result.reference_keys])
    aligned_points = np.vstack([aligned_mobile.atom(key).xyz for key in result.mobile_keys])
    _add_local_motion(result, reference_points, aligned_points)
    return result


def _add_local_motion(result, reference_points, aligned_points):
    reference_center = reference_points.mean(axis=0)
    aligned_center = aligned_points.mean(axis=0)
    translation = reference_center - aligned_center
    result.local_translation = translation.tolist()
    if len(reference_points) < 3:
        residual = (aligned_points + translation) - reference_points
        result.internal_vectors = residual.tolist()
        result.internal_distances = np.linalg.norm(residual, axis=1).tolist()
        result.internal_rmsd_value = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        result.rigid_fit_status = "underdetermined_rotation"
        return

    reference_centered = reference_points - reference_center
    aligned_centered = aligned_points - aligned_center
    rank = int(np.linalg.matrix_rank(np.vstack([reference_centered, aligned_centered]), tol=1e-8))
    if rank < 2:
        result.rigid_fit_status = "degenerate_geometry"
        return

    covariance = aligned_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = vt.T @ u.T
    translation = reference_center - rotation @ aligned_center
    result.local_translation = translation.tolist()
    fitted = (aligned_points - aligned_center) @ rotation.T + reference_center
    residual = fitted - reference_points
    angle_value, axis = _rotation_axis_angle(rotation)
    result.local_rotation = rotation.tolist()
    result.local_rotation_angle = angle_value
    result.local_rotation_axis = axis if isinstance(axis, list) else axis.tolist()
    result.internal_vectors = residual.tolist()
    result.internal_distances = np.linalg.norm(residual, axis=1).tolist()
    result.internal_rmsd_value = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    result.rigid_fit_status = "fitted"


def _rotation_axis_angle(rotation):
    trace_value = min(3.0, max(-1.0, float(np.trace(rotation))))
    cosine = min(1.0, max(-1.0, (trace_value - 1.0) / 2.0))
    angle_value = float(np.degrees(np.arccos(cosine)))
    if angle_value < 1e-10 or angle_value > 180.0 - 1e-10:
        return angle_value, [0.0, 0.0, 1.0]
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return angle_value, (axis / np.linalg.norm(axis)).tolist()
