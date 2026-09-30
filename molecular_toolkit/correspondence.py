from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .structures import Atom, Structure


AtomKey = int
IdentityMode = str
Pair = tuple[AtomKey, AtomKey]
AtomSelector = AtomKey | tuple[object, ...] | Mapping[str, object]


@dataclass(eq=False)
class AtomCorrespondence:
    pairs: list[Pair]
    unmatched_reference: list[AtomKey] = field(default_factory=list)
    unmatched_mobile: list[AtomKey] = field(default_factory=list)
    ambiguous_identities: dict[str, dict[str, list[AtomKey]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    confirmed: bool = False
    confirmation_note: str | None = None

    @property
    def is_complete(self) -> bool:
        return not self.unmatched_reference and not self.unmatched_mobile and not self.ambiguous_identities

    def pair_count(self) -> int:
        return len(self.pairs)


def match_atoms(
    reference: Structure,
    mobile: Structure,
    explicit_pairs: Sequence[tuple[AtomSelector, AtomSelector]] | Mapping[AtomSelector, AtomSelector] | None = None,
    identity: IdentityMode = "serial",
    require_same_element: bool = True,
) -> AtomCorrespondence:
    if explicit_pairs is not None:
        return match_explicit(reference, mobile, explicit_pairs, require_same_element)
    if identity == "none":
        return AtomCorrespondence(
            pairs=[],
            unmatched_reference=list(range(len(reference))),
            unmatched_mobile=list(range(len(mobile))),
            notes=["Automatic matching is disabled; provide explicit_pairs."],
        )
    return match_by_identity(reference, mobile, identity, require_same_element)


def confirm_correspondence(
    correspondence: AtomCorrespondence,
    note: str = "Human review confirmed this atom correspondence.",
) -> AtomCorrespondence:
    """Record a human decision while retaining unmatched and ambiguous atoms."""
    correspondence.confirmed = True
    correspondence.confirmation_note = note
    if "human_confirmed" not in correspondence.notes:
        correspondence.notes.append("human_confirmed")
    return correspondence


def match_explicit(
    reference: Structure,
    mobile: Structure,
    explicit_pairs: Sequence[tuple[AtomSelector, AtomSelector]] | Mapping[AtomSelector, AtomSelector],
    require_same_element: bool,
) -> AtomCorrespondence:
    if isinstance(explicit_pairs, Mapping):
        raw_pairs = list(explicit_pairs.items())
    else:
        raw_pairs = list(explicit_pairs)
    pairs: list[Pair] = []
    used_reference: set[AtomKey] = set()
    used_mobile: set[AtomKey] = set()
    notes: list[str] = ["manual_correspondence"]
    for ref_selector, mobile_selector in raw_pairs:
        ref_key = resolve_atom_key(reference, ref_selector, "reference")
        mobile_key = resolve_atom_key(mobile, mobile_selector, "mobile")
        if ref_key in used_reference:
            raise ValueError(f"Reference atom key {ref_key} appears in more than one explicit pair")
        if mobile_key in used_mobile:
            raise ValueError(f"Mobile atom key {mobile_key} appears in more than one explicit pair")
        if require_same_element and reference.atom(ref_key).element != mobile.atom(mobile_key).element:
            raise ValueError(
                f"Element mismatch in explicit pair: {reference.atom(ref_key).element} != "
                f"{mobile.atom(mobile_key).element}"
            )
        used_reference.add(ref_key)
        used_mobile.add(mobile_key)
        pairs.append((ref_key, mobile_key))
    if len(pairs) < 3:
        notes.append("At least three non-collinear matched atoms are required for rigid alignment.")
    return AtomCorrespondence(
        pairs=pairs,
        unmatched_reference=[key for key in range(len(reference)) if key not in used_reference],
        unmatched_mobile=[key for key in range(len(mobile)) if key not in used_mobile],
        notes=notes,
        confirmed=True,
        confirmation_note="All supplied atom pairs were explicitly selected.",
    )


def resolve_atom_key(structure: Structure, selector: AtomSelector, side: str) -> AtomKey:
    if isinstance(selector, int) and not isinstance(selector, bool):
        structure.atom(selector)
        return selector
    criteria: Mapping[str, object]
    if isinstance(selector, Mapping):
        criteria = dict(selector)
    elif isinstance(selector, tuple):
        if selector and selector[0] == "key":
            key = int(selector[1])
            structure.atom(key)
            return key
        if selector and selector[0] == "serial":
            criteria = {"serial": selector[1]}
        elif len(selector) == 3:
            criteria = {"chain": selector[0], "residue_seq": selector[1], "name": selector[2]}
        else:
            raise ValueError(f"Unsupported {side} atom selector: {selector!r}")
    else:
        raise ValueError(f"Unsupported {side} atom selector: {selector!r}")
    matches = [atom.key for atom in structure.atoms if atom_matches(atom, criteria)]
    if not matches:
        raise ValueError(f"No {side} atom matches {criteria}")
    if len(matches) > 1:
        raise ValueError(f"{side.capitalize()} selector {criteria} matches multiple atoms: {matches}")
    return matches[0]


def atom_matches(atom: Atom, criteria: Mapping[str, object]) -> bool:
    return all(getattr(atom, name) == value for name, value in criteria.items())


def match_by_identity(
    reference: Structure,
    mobile: Structure,
    identity: IdentityMode,
    require_same_element: bool,
) -> AtomCorrespondence:
    ref_index, ref_ambiguous = identity_index(reference, identity)
    mob_index, mob_ambiguous = identity_index(mobile, identity)
    ambiguous = combine_ambiguous(reference, mobile, identity, ref_ambiguous, mob_ambiguous)
    pairs: list[Pair] = []
    matched_ref: set[AtomKey] = set()
    matched_mob: set[AtomKey] = set()
    notes: list[str] = []
    for identity_value, ref_keys in ref_index.items():
        mobile_keys = mob_index.get(identity_value)
        if not mobile_keys or identity_value in ref_ambiguous or identity_value in mob_ambiguous:
            continue
        ref_key = ref_keys[0]
        mobile_key = mobile_keys[0]
        if require_same_element and reference.atom(ref_key).element != mobile.atom(mobile_key).element:
            notes.append(f"Identity {identity_value!r} skipped because elements differ")
            continue
        pairs.append((ref_key, mobile_key))
        matched_ref.add(ref_key)
        matched_mob.add(mobile_key)
    if len(pairs) < 3:
        notes.append("At least three non-collinear matched atoms are required for rigid alignment.")
    return AtomCorrespondence(
        pairs=pairs,
        unmatched_reference=[key for key in range(len(reference)) if key not in matched_ref],
        unmatched_mobile=[key for key in range(len(mobile)) if key not in matched_mob],
        ambiguous_identities=ambiguous,
        notes=notes,
    )


def identity_index(structure: Structure, identity: IdentityMode) -> tuple[dict[object, list[AtomKey]], set[object]]:
    grouped: dict[object, list[AtomKey]] = defaultdict(list)
    for atom in structure.atoms:
        value = atom_identity(atom, identity)
        if value is not None:
            grouped[value].append(atom.key)
    unique = {value: keys for value, keys in grouped.items() if len(keys) == 1}
    ambiguous = {value for value, keys in grouped.items() if len(keys) > 1}
    return unique, ambiguous


def atom_identity(atom: Atom, identity: IdentityMode) -> object | None:
    if identity == "serial":
        return None if atom.serial is None else ("serial", atom.serial)
    if identity == "pdb_atom":
        values = (atom.chain, atom.residue_seq, atom.name)
        return None if any(value is None for value in values) else ("pdb_atom",) + values
    raise ValueError(f"Unsupported identity mode: {identity}")


def combine_ambiguous(
    reference: Structure,
    mobile: Structure,
    identity: IdentityMode,
    ref_ambiguous: set[object],
    mob_ambiguous: set[object],
) -> dict[str, dict[str, list[AtomKey]]]:
    result: dict[str, dict[str, list[AtomKey]]] = {}
    for value in sorted(ref_ambiguous | mob_ambiguous, key=repr):
        result[repr_identity(value)] = {
            "reference": grouped_keys(reference, value, identity),
            "mobile": grouped_keys(mobile, value, identity),
        }
    return result


def grouped_keys(structure: Structure, value: object, identity: IdentityMode) -> list[AtomKey]:
    return [atom.key for atom in structure.atoms if atom_identity(atom, identity) == value]


def repr_identity(value: object) -> str:
    if isinstance(value, tuple):
        return ":".join(str(part) for part in value[1:])
    return str(value)
