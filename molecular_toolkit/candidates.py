from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Mapping, Sequence

from .correspondence import AtomCorrespondence, Pair, match_atoms, resolve_atom_key
from .structures import Structure


@dataclass(eq=False)
class CorrespondenceCandidate:
    correspondence: AtomCorrespondence
    locked_pairs: list[Pair] = field(default_factory=list)
    ambiguous_groups: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    is_mirror: bool = False
    score: float | None = None
    label: str = "candidate"


def build_correspondence_candidates(
    reference: Structure,
    mobile: Structure,
    locked_pairs: Sequence[tuple[object, object]] | Mapping[object, object] | None = None,
    ambiguous_groups: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    identity: str = "serial",
    include_mirror: bool = False,
    require_same_element: bool = True,
) -> list[CorrespondenceCandidate]:
    locked = normalize_pairs(reference, mobile, locked_pairs)
    groups = normalize_groups(ambiguous_groups)
    choices: list[list[Pair]] = []
    reported_groups: dict[str, dict[str, list[int]]] = {}
    used_reference = {ref_key for ref_key, _ in locked}
    used_mobile = {mobile_key for _, mobile_key in locked}

    for group_name, group in groups.items():
        ref_keys = [int(key) for key in group["reference"]]
        mobile_keys = [int(key) for key in group["mobile"]]
        validate_group(group_name, ref_keys, mobile_keys, used_reference, used_mobile)
        reported_groups[group_name] = {"reference": ref_keys, "mobile": mobile_keys}
        choices.append(permutation_choices(group_name, ref_keys, mobile_keys))
        used_reference.update(ref_keys)
        used_mobile.update(mobile_keys)

    base = match_atoms(reference, mobile, identity=identity, require_same_element=require_same_element)
    fixed_pairs = locked + [
        pair
        for pair in base.pairs
        if pair[0] not in used_reference and pair[1] not in used_mobile
    ]
    used_reference.update(pair[0] for pair in fixed_pairs)
    used_mobile.update(pair[1] for pair in fixed_pairs)
    unmatched_reference = [key for key in range(len(reference)) if key not in used_reference]
    unmatched_mobile = [key for key in range(len(mobile)) if key not in used_mobile]

    candidates: list[CorrespondenceCandidate] = []
    combinations = product(*choices) if choices else [()]
    for index, chosen in enumerate(combinations):
        candidate_pairs = fixed_pairs + [pair for group_assignment in chosen for pair in group_assignment]
        correspondence = AtomCorrespondence(
            pairs=candidate_pairs,
            unmatched_reference=unmatched_reference,
            unmatched_mobile=unmatched_mobile,
            ambiguous_identities=reported_groups,
            notes=["Built from locked pairs and enumerated ambiguous groups."],
        )
        candidates.append(
            CorrespondenceCandidate(
                correspondence=correspondence,
                locked_pairs=locked,
                ambiguous_groups=reported_groups,
                label=f"candidate_{index}",
            )
        )

    if include_mirror:
        mirrored = [
            CorrespondenceCandidate(
                correspondence=candidate.correspondence,
                locked_pairs=candidate.locked_pairs,
                ambiguous_groups=candidate.ambiguous_groups,
                is_mirror=True,
                label=candidate.label + "_mirror",
            )
            for candidate in candidates
        ]
        candidates.extend(mirrored)
    return candidates


def lock_candidate_pairs(
    candidate: CorrespondenceCandidate,
    locked_pairs: Sequence[Pair] | Mapping[int, int],
    reference: Structure | None = None,
    mobile: Structure | None = None,
) -> CorrespondenceCandidate:
    normalized = (
        normalize_integer_pairs(locked_pairs)
        if reference is None or mobile is None
        else normalize_pairs(reference, mobile, locked_pairs)
    )
    locked_reference = {pair[0] for pair in normalized}
    locked_mobile = {pair[1] for pair in normalized}
    pairs = {
        pair
        for pair in candidate.correspondence.pairs
        if pair[0] not in locked_reference and pair[1] not in locked_mobile
    }
    for pair in normalized:
        pairs.add(pair)
    existing_locked = {
        pair
        for pair in candidate.locked_pairs
        if pair[0] not in locked_reference and pair[1] not in locked_mobile
    }
    locked = sorted(existing_locked.union(normalized))
    reference_keys = (
        set(range(len(reference)))
        if reference is not None
        else set(candidate.correspondence.unmatched_reference) | {pair[0] for pair in pairs}
    )
    mobile_keys = (
        set(range(len(mobile)))
        if mobile is not None
        else set(candidate.correspondence.unmatched_mobile) | {pair[1] for pair in pairs}
    )
    correspondence = AtomCorrespondence(
        pairs=sorted(pairs),
        unmatched_reference=sorted(reference_keys - {pair[0] for pair in pairs}),
        unmatched_mobile=sorted(mobile_keys - {pair[1] for pair in pairs}),
        ambiguous_identities=candidate.correspondence.ambiguous_identities,
        notes=candidate.correspondence.notes + ["Additional relationships were manually locked."],
    )
    return CorrespondenceCandidate(
        correspondence=correspondence,
        locked_pairs=locked,
        ambiguous_groups=candidate.ambiguous_groups,
        is_mirror=candidate.is_mirror,
        score=candidate.score,
        label=candidate.label,
    )


def normalize_pairs(
    reference: Structure,
    mobile: Structure,
    pairs: Sequence[tuple[object, object]] | Mapping[object, object] | None,
) -> list[Pair]:
    if pairs is None:
        return []
    raw = pairs.items() if isinstance(pairs, Mapping) else pairs
    normalized = [
        (
            resolve_atom_key(reference, ref_selector, "reference"),
            resolve_atom_key(mobile, mobile_selector, "mobile"),
        )
        for ref_selector, mobile_selector in raw
    ]
    if len({pair[0] for pair in normalized}) != len(normalized):
        raise ValueError("Locked pairs cannot reuse a reference atom")
    if len({pair[1] for pair in normalized}) != len(normalized):
        raise ValueError("Locked pairs cannot reuse a mobile atom")
    return normalized


def normalize_integer_pairs(pairs: Sequence[Pair] | Mapping[int, int]) -> list[Pair]:
    raw = pairs.items() if isinstance(pairs, Mapping) else pairs
    normalized = [(int(ref_key), int(mobile_key)) for ref_key, mobile_key in raw]
    if len(set(pair[0] for pair in normalized)) != len(normalized):
        raise ValueError("Locked pairs cannot reuse a reference atom")
    if len(set(pair[1] for pair in normalized)) != len(normalized):
        raise ValueError("Locked pairs cannot reuse a mobile atom")
    return normalized


def normalize_groups(
    groups: Mapping[str, Mapping[str, Sequence[int]]] | None,
) -> dict[str, dict[str, list[int]]]:
    if groups is None:
        return {}
    result = {}
    for name, group in groups.items():
        result[str(name)] = {
            "reference": [int(key) for key in group["reference"]],
            "mobile": [int(key) for key in group["mobile"]],
        }
    return result


def validate_group(
    name: str,
    reference_keys: list[int],
    mobile_keys: list[int],
    used_reference: set[int],
    used_mobile: set[int],
) -> None:
    if not reference_keys or not mobile_keys:
        raise ValueError(f"Ambiguous group {name!r} must contain atoms on both sides")
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError(f"Ambiguous group {name!r} contains repeated reference atoms")
    if len(set(mobile_keys)) != len(mobile_keys):
        raise ValueError(f"Ambiguous group {name!r} contains repeated mobile atoms")
    overlap_reference = sorted(used_reference.intersection(reference_keys))
    overlap_mobile = sorted(used_mobile.intersection(mobile_keys))
    if overlap_reference or overlap_mobile:
        raise ValueError(f"Ambiguous group {name!r} overlaps a locked relationship")


def permutation_choices(group_name: str, reference_keys: list[int], mobile_keys: list[int]) -> list[Pair]:
    if len(reference_keys) != len(mobile_keys):
        raise ValueError(
            f"Ambiguous group {name!r} must contain equal reference and mobile atom counts; "+
            "represent missing atoms by leaving them unmatched instead"
        )
    if len(reference_keys) == 1:
        return (((reference_keys[0], mobile_keys[0]),),)
    return [
        tuple((ref_key, mobile_keys[mobile_index]) for ref_key, mobile_index in zip(reference_keys, order))
            for order in permutations(range(len(mobile_keys)))]
