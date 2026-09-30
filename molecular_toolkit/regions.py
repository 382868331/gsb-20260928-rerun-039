from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .correspondence import Pair
from .geometry import ReflectionAlignmentError, kabsch_transforms
from .motion import LocalMotion, decompose_local_motion
from .structures import Structure, transform_structure


AtomKeys = Iterable[int]


@dataclass(eq=False)
class RegionDisplacements:
    name: str
    reference_keys: list[int]
    mobile_keys: list[int]
    distances: list[float] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    motion: LocalMotion | None = None

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
            "motion": self.motion.to_dict() if self.motion is not None else None,
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
    )

    assigned_pairs: set[Pair] = set(stable_pairs)
    if regions:
        for name, value in regions.items():
            ref_keys, mobile_keys = _normalize_region(value)
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
    measured = RegionDisplacements(
        name=name,
        reference_keys=[pair[0] for pair in pairs],
        mobile_keys=[pair[1] for pair in pairs],
        distances=distances,
        vectors=vectors,
    )
    reference_points = np.vstack([reference.atom(pair[0]).xyz for pair in pairs])
    aligned_points = np.vstack([aligned_mobile.atom(pair[1]).xyz for pair in pairs])
    measured.motion = decompose_local_motion(reference_points, aligned_points)
    return measured
