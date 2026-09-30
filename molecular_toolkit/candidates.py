from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .correspondence import AtomCorrespondence, Pair, match_atoms
from .geometry import kabsch_transforms
from .structures import Structure


@dataclass(eq=False, order=False)
class CorrespondenceCandidate:
    candidate_id: str
    pairs: list[Pair]
    locked_pairs: list[Pair] = field(default_factory=list)
    unmatched_reference: list[int] = field(default_factory=list)
    unmatched_mobile: list[int] = field(default_factory=list)
    ambiguous_identities: dict = field(default_factory=dict)
    rmsd_proper: float | None = None
    rmsd_reflected: float | None = None
    is_mirror_candidate: bool = False
    fit_error: str | None = None
    confirmed: bool = False
    confirmation_note: str | None = None

    @property
    def rmsd(self) -> float | None:
        return self.rmsd_reflected if self.is_mirror_candidate else self.rmsd_proper

    def to_correspondence(self) -> AtomCorrespondence:
        locked = set(self.locked_pairs)
        notes = [
            f"candidate={self.candidate_id}",
            f"locked_pairs={len(locked)}",
        ]
        confirmed = self.confirmed or bool(locked)
        if confirmed:
            notes.append("human_confirmed")
        return AtomCorrespondence(
            pairs=list(self.pairs),
            unmatched_reference=list(self.unmatched_reference),
            unmatched_mobile=list(self.unmatched_mobile),
            ambiguous_identities=dict(self.ambiguous_identities),
            notes=notes,
            confirmed=confirmed,
            confirmation_note=self.confirmation_note
            or ("At least one equivalent-atom relation was manually locked." if locked else None),
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "pairs": [
                {"reference_key": ref_key, "mobile_key": mobile_key}
                for ref_key, mobile_key in self.pairs
            ],
            "unmatched_reference": self.unmatched_reference,
            "unmatched_mobile": self.unmatched_mobile,
            "ambiguous_identities": self.ambiguous_identities,
            "locked_pairs": [
                {"reference_key": ref_key, "mobile_key": mobile_key}
                for ref_key, mobile_key in self.locked_pairs
            ],
            "rmsd_proper": self.rmsd_proper,
            "rmsd_reflected": self.rmsd_reflected,
            "is_mirror_candidate": self.is_mirror_candidate,
            "fit_error": self.fit_error,
            "confirmed": self.confirmed or bool(self.locked_pairs),
            "confirmation_note": self.confirmation_note
            or ("At least one equivalent-atom relation was manually locked." if self.locked_pairs else None),
        }


def confirm_candidate(
    candidate: CorrespondenceCandidate,
    note: str = "Human review selected this correspondence candidate.",
) -> CorrespondenceCandidate:
    candidate.confirmed = True
    candidate.confirmation_note = note
    return candidate


def enumerate_candidates(
    reference: Structure,
    mobile: Structure,
    ambiguous_groups: Sequence[tuple[Sequence[int], Sequence[int]]] | Mapping[str, tuple[Sequence[int], Sequence[int]]],
    *,
    locked_pairs: Sequence[Pair] | Mapping[int, int] | None = None,
    base_pairs: Sequence[Pair] | None = None,
    identity: str = "serial",
    require_same_element: bool = True,
    fit: bool = True,
    max_candidates: int = 256,
) -> list[CorrespondenceCandidate]:
    """Enumerate bijections for ambiguous symmetric/duplicate atom groups.

    Each group is ``(reference_keys, mobile_keys)``. Equal-size groups become
    permutations; unequal groups are not forced and their atoms remain outside
    the candidate pairs. ``locked_pairs`` removes both atoms from enumeration.
    ``base_pairs`` provides certain pairs; by default unique serial matches are
    used. Every returned candidate keeps the same locked relation.
    """
    locked = _normalize_pairs(locked_pairs)
    _validate_locked(reference, mobile, locked, require_same_element)
    locked_ref = {ref for ref, _ in locked}
    locked_mob = {mob for _, mob in locked}

    base = _base_pairs(reference, mobile, base_pairs, identity, require_same_element)
    for ref_key, mobile_key in base:
        if ref_key in locked_ref and mobile_key != dict(locked).get(ref_key):
            raise ValueError(f"Locked reference atom {ref_key} conflicts with base pair")
        if mobile_key in locked_mob and (ref_key, mobile_key) not in locked:
            raise ValueError(f"Locked mobile atom {mobile_key} conflicts with base pair")
    base = [(ref, mob) for ref, mob in base if ref not in locked_ref and mob not in locked_mob]

    groups = _normalize_groups(ambiguous_groups)
    group_ambiguity = {
        f"group-{index}": {"reference": list(ref_keys), "mobile": list(mob_keys)}
        for index, (ref_keys, mob_keys) in enumerate(groups)
    }
    option_lists: list[list[list[Pair]]] = []
    for index, (ref_keys, mob_keys) in enumerate(groups):
        ref_keys, mob_keys = _filter_group(ref_keys, mob_keys, locked_ref, locked_mob, index)
        if not ref_keys and not mob_keys:
            continue
        if len(ref_keys) != len(mob_keys):
            raise ValueError(
                f"Ambiguous group {index} has {len(ref_keys)} reference and {len(mob_keys)} "
                "unlocked mobile atoms; lock pairs or leave both sides unmatched"
            )
        _validate_group(reference, mobile, ref_keys, mob_keys, require_same_element)
        permutations = [list(zip(ref_keys, permutation)) for permutation in itertools.permutations(mob_keys)]
        option_lists.append(permutations)

    combinations = list(itertools.product(*option_lists)) if option_lists else [()]
    if len(combinations) > max_candidates:
        raise ValueError(
            f"{len(combinations)} candidates exceed max_candidates={max_candidates}; "
            "lock more ambiguous atom pairs"
        )

    candidates = []
    for index, combination in enumerate(combinations):
        extra = [pair for permutation in combination for pair in permutation]
        pairs = _merge_pairs(base, locked, extra)
        candidate = CorrespondenceCandidate(
            candidate_id=f"cand-{index:03d}",
            pairs=pairs,
            locked_pairs=list(locked),
            ambiguous_identities=dict(group_ambiguity),
            unmatched_reference=[key for key in range(len(reference)) if key not in {ref for ref, _ in pairs}],
            unmatched_mobile=[key for key in range(len(mobile)) if key not in {mob for _, mob in pairs}],
        )
        if fit:
            _score_candidate(reference, mobile, candidate)
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item.rmsd is None, item.rmsd if item.rmsd is not None else 0.0))
    return candidates


def lock_candidate_pairs(
    candidates: Sequence[CorrespondenceCandidate], locked_pairs: Sequence[Pair]
) -> list[CorrespondenceCandidate]:
    """Keep only candidates consistent with additional human-locked relations."""
    required = set(_normalize_pairs(locked_pairs))
    filtered = []
    for candidate in candidates:
        current = set(candidate.pairs)
        if required <= current:
            candidate.locked_pairs = sorted(set(candidate.locked_pairs) | required)
            filtered.append(candidate)
    return filtered


def candidates_from_identity(
    reference: Structure,
    mobile: Structure,
    *,
    identity: str = "serial",
    locked_pairs: Sequence[Pair] | None = None,
    require_same_element: bool = True,
    fit: bool = True,
    max_candidates: int = 256,
) -> list[CorrespondenceCandidate]:
    """Build candidates from duplicate/missing identities reported by match_atoms."""
    correspondence = match_atoms(
        reference, mobile, identity=identity, require_same_element=require_same_element
    )
    groups = []
    for item in correspondence.ambiguous_identities.values():
        ref_keys = item.get("reference", [])
        mob_keys = item.get("mobile", [])
        if ref_keys and mob_keys:
            groups.append((ref_keys, mob_keys))
    return enumerate_candidates(
        reference,
        mobile,
        groups,
        locked_pairs=locked_pairs,
        base_pairs=correspondence.pairs,
        identity=identity,
        require_same_element=require_same_element,
        fit=fit,
        max_candidates=max_candidates,
    )


def _score_candidate(reference, mobile, candidate):
    try:
        ref_points = np.vstack([reference.atom(ref).xyz for ref, _ in candidate.pairs])
        mob_points = np.vstack([mobile.atom(mob).xyz for _, mob in candidate.pairs])
        _, _, _, rmsd_proper, rmsd_reflected = kabsch_transforms(ref_points, mob_points)
        candidate.rmsd_proper = float(rmsd_proper)
        candidate.rmsd_reflected = float(rmsd_reflected)
        candidate.is_mirror_candidate = bool(rmsd_reflected < rmsd_proper)
    except (ValueError, np.linalg.LinAlgError) as exc:
        candidate.fit_error = str(exc)


def _base_pairs(reference, mobile, base_pairs, identity, require_same_element):
    if base_pairs is not None:
        return _normalize_pairs(base_pairs)
    correspondence = match_atoms(
        reference, mobile, identity=identity, require_same_element=require_same_element
    )
    return list(correspondence.pairs)


def _normalize_pairs(pairs):
    if pairs is None:
        return []
    if isinstance(pairs, Mapping):
        pairs = list(pairs.items())
    normalized = [(int(ref), int(mob)) for ref, mob in pairs]
    if len({ref for ref, _ in normalized}) != len(normalized):
        raise ValueError("Reference atom appears in multiple locked/base pairs")
    if len({mob for _, mob in normalized}) != len(normalized):
        raise ValueError("Mobile atom appears in multiple locked/base pairs")
    return normalized


def _validate_locked(reference, mobile, locked, require_same_element):
    for ref_key, mobile_key in locked:
        reference.atom(ref_key)
        mobile.atom(mobile_key)
        if require_same_element and reference.atom(ref_key).element != mobile.atom(mobile_key).element:
            raise ValueError(
                f"Locked pair ({ref_key}, {mobile_key}) has element mismatch: "
                f"{reference.atom(ref_key).element} != {mobile.atom(mobile_key).element}"
            )


def _normalize_groups(groups):
    if isinstance(groups, Mapping):
        return [(list(value[0]), list(value[1])) for value in groups.values()]
    normalized = []
    for value in groups:
        if len(value) != 2:
            raise ValueError("Ambiguous group must contain reference and mobile key lists")
        normalized.append((list(value[0]), list(value[1])))
    return normalized


def _filter_group(ref_keys, mob_keys, locked_ref, locked_mob, index):
    refs = [key for key in ref_keys if key not in locked_ref]
    mobs = [key for key in mob_keys if key not in locked_mob]
    locked_refs = {key for key in ref_keys if key in locked_ref}
    locked_mobs = {key for key in mob_keys if key in locked_mob}
    if bool(locked_refs) != bool(locked_mobs):
        raise ValueError(f"Ambiguous group {index} is locked on only one structure")
    return refs, mobs


def _validate_group(reference, mobile, ref_keys, mob_keys, require_same_element):
    if len(set(ref_keys)) != len(ref_keys) or len(set(mob_keys)) != len(mob_keys):
        raise ValueError("Ambiguous groups contain duplicate atom keys")
    for key in ref_keys:
        reference.atom(key)
    for key in mob_keys:
        mobile.atom(key)
    if require_same_element:
        ref_elements = sorted(reference.atom(key).element for key in ref_keys)
        mob_elements = sorted(mobile.atom(key).element for key in mob_keys)
        if ref_elements != mob_elements:
            raise ValueError("Ambiguous group elements differ between reference and mobile")


def _merge_pairs(base, locked, extra):
    pairs = list(base) + list(locked) + list(extra)
    refs = [ref for ref, _ in pairs]
    mobs = [mob for _, mob in pairs]
    if len(set(refs)) != len(refs) or len(set(mobs)) != len(mobs):
        raise ValueError("Candidate contains conflicting pairs")
    return pairs
