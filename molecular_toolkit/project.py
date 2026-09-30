from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .correspondence import AtomCorrespondence, Pair, match_atoms
from .ensemble import Conformer, ConformerComparison, compare_ensemble
from .structures import load_structure


PROJECT_VERSION = 1


@dataclass(eq=False)
class RegionDefinition:
    reference_keys: list[int]
    mobile_keys: list[int] | None = None

    def to_dict(self) -> dict:
        return {"reference_keys": self.reference_keys, "mobile_keys": self.mobile_keys}

    @classmethod
    def from_dict(cls, data: dict) -> "RegionDefinition":
        return cls(
            reference_keys=list(data["reference_keys"]),
            mobile_keys=None if data.get("mobile_keys") is None else list(data["mobile_keys"]),
        )


class Project:
    """Persistable comparison setup: manual correspondences, reference and regions.

    Only original file locations, human decisions (pairs, locks, selected
    stable atoms, region definitions) and computed results are stored.
    Original coordinates are never rewritten; reopening reloads the same
    source files (verified by a content hash) and recomputes identical
    measurements.
    """

    def __init__(
        self,
        reference: Conformer,
        conformers: Sequence[Conformer],
        *,
        name: str = "comparison_project",
    ):
        self.name = name
        self.reference = reference
        self.conformers: list[Conformer] = list(conformers)
        self.locked_pairs: dict[int, list[Pair]] = {}
        self.manual_pairs: dict[int, list[Pair]] = {}
        self.pending_alternatives: dict[int, list[list[Pair]]] = {}
        self.stable_reference_keys: list[int] = []
        self.regions: dict[str, RegionDefinition] = {}
        self.comparisons: list[ConformerComparison] | None = None
        self.notes: list[str] = []
        self._check_hashes()

    @classmethod
    def from_files(
        cls,
        reference_path: str | Path,
        conformer_paths: Sequence[str | Path],
        *,
        name: str = "comparison_project",
    ) -> "Project":
        reference = Conformer.from_file(reference_path)
        conformers = [Conformer.from_file(path) for path in conformer_paths]
        return cls(reference, conformers, name=name)

    def conformer_index(self, conformer: int | str) -> int:
        if isinstance(conformer, int):
            if not 0 <= conformer < len(self.conformers):
                raise IndexError(f"Unknown conformer index {conformer}")
            return conformer
        names = [item.name for item in self.conformers]
        if names.count(conformer) != 1:
            raise ValueError(f"Conformer name {conformer!r} is missing or not unique")
        return names.index(conformer)

    def set_manual_pairs(
        self,
        conformer: int | str,
        pairs: Sequence[Pair],
        *,
        require_same_element: bool = True,
    ) -> None:
        """Record a human-confirmed correspondence (including symmetric swaps)."""
        index = self.conformer_index(conformer)
        normalized = match_atoms(
            self.reference.structure,
            self.conformers[index].structure,
            explicit_pairs=list(pairs),
            require_same_element=require_same_element,
        ).pairs
        self.manual_pairs[index] = normalized

    def add_alternative_pairs(
        self,
        conformer: int | str,
        alternatives: Sequence[Sequence[Pair]],
    ) -> None:
        """Keep optional bijections (e.g. equivalent-atom exchanges) pending review.

        Alternatives are not merged into the active mapping until
        ``confirm_alternative`` is called, so ambiguous interpretations stay
        visible instead of being silently resolved.
        """
        index = self.conformer_index(conformer)
        stored = []
        for option in alternatives:
            stored.append([(int(ref), int(mob)) for ref, mob in option])
        self.pending_alternatives[index] = stored

    def confirm_alternative(self, conformer: int | str, option_index: int) -> None:
        index = self.conformer_index(conformer)
        options = self.pending_alternatives.get(index)
        if not options or not 0 <= option_index < len(options):
            raise ValueError("No pending alternative with that index")
        chosen = options[option_index]
        self.set_manual_pairs(index, chosen)
        self.locked_pairs[index] = list(chosen)

    def lock_pairs(self, conformer: int | str, pairs: Sequence[Pair]) -> None:
        index = self.conformer_index(conformer)
        locked = [(int(ref), int(mob)) for ref, mob in pairs]
        self.locked_pairs[index] = locked

    def set_stable_region(self, reference_keys: Iterable[int]) -> None:
        keys = list(reference_keys)
        if len(keys) != len(set(keys)):
            raise ValueError("Stable reference keys must be unique")
        for key in keys:
            self.reference.structure.atom(key)
        self.stable_reference_keys = keys

    def set_region(
        self,
        name: str,
        reference_keys: Iterable[int],
        mobile_keys: Iterable[int] | None = None,
    ) -> None:
        self.regions[name] = RegionDefinition(list(reference_keys), None if mobile_keys is None else list(mobile_keys))

    def correspondence_for(self, index: int) -> AtomCorrespondence:
        if index in self.manual_pairs:
            pairs = self.manual_pairs[index]
        else:
            pairs = match_atoms(
                self.reference.structure, self.conformers[index].structure
            ).pairs
        return AtomCorrespondence(
            pairs=pairs,
            notes=["manual" if index in self.manual_pairs else "identity=serial"],
        )

    def run(self) -> list[ConformerComparison]:
        if len(self.stable_reference_keys) < 3:
            raise ValueError("Select at least three stable reference atoms before running")
        pairs = [self.correspondence_for(index) for index in range(len(self.conformers))]
        regions = {name: (definition.reference_keys, definition.mobile_keys) for name, definition in self.regions.items()}
        self.comparisons = compare_ensemble(
            self.reference,
            list(zip(self.conformers, pairs)),
            stable_reference_keys=self.stable_reference_keys,
            regions=regions,
        )
        return self.comparisons

    def to_dict(self) -> dict:
        self._check_hashes()
        conformer_records = []
        for index, conformer in enumerate(self.conformers):
            conformer_records.append(
                {
                    "provenance": conformer.provenance(),
                    "sha256": file_hash(Path(conformer.source_path)),
                    "locked_pairs": _pairs_to_dict(self.locked_pairs.get(index, [])),
                    "manual_pairs": _pairs_to_dict(self.manual_pairs.get(index, [])),
                    "pending_alternatives": [
                        _pairs_to_dict(option)
                        for option in self.pending_alternatives.get(index, [])
                    ],
                }
            )
        return {
            "toolkit": "molecular_toolkit",
            "project_version": PROJECT_VERSION,
            "name": self.name,
            "reference": {
                "provenance": self.reference.provenance(),
                "sha256": file_hash(Path(self.reference.source_path)),
            },
            "stable_reference_keys": self.stable_reference_keys,
            "regions": {name: definition.to_dict() for name, definition in self.regions.items()},
            "conformers": conformer_records,
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, rerun: bool = True) -> "Project":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("project_version") != PROJECT_VERSION:
            raise ValueError("Unsupported project version")
        reference = Conformer.from_file(data["reference"]["provenance"]["source_path"])
        _verify_hash(reference, data["reference"])
        conformers = []
        for record in data["conformers"]:
            conformer = Conformer.from_file(record["provenance"]["source_path"])
            _verify_hash(conformer, record)
            conformers.append(conformer)
        project = cls(reference, conformers, name=data.get("name", "comparison_project"))
        project.stable_reference_keys = list(data.get("stable_reference_keys", []))
        project.regions = {
            name: RegionDefinition.from_dict(definition)
            for name, definition in data.get("regions", {}).items()
        }
        project.notes = list(data.get("notes", []))
        for index, record in enumerate(data["conformers"]):
            project.manual_pairs[index] = _pairs_from_dict(record.get("manual_pairs", []))
            project.locked_pairs[index] = _pairs_from_dict(record.get("locked_pairs", []))
            project.pending_alternatives[index] = [
                _pairs_from_dict(option) for option in record.get("pending_alternatives", [])
            ]
        if rerun and project.stable_reference_keys:
            project.run()
        return project

    def _check_hashes(self) -> None:
        for conformer in [self.reference, *self.conformers]:
            if conformer.source_path is None or not Path(conformer.source_path).exists():
                raise FileNotFoundError(
                    f"Source file required for reproducibility is unavailable: {conformer.source_path}"
                )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(conformer: Conformer, record: dict) -> None:
    expected = record.get("sha256")
    if expected is None:
        return
    actual = file_hash(Path(conformer.source_path))
    if actual != expected:
        raise ValueError(
            f"Source file for {conformer.name!r} changed since the project was saved; "
            "manual correspondences and measurements are no longer reproducible."
        )


def _pairs_to_dict(pairs: Sequence[Pair]) -> list[dict]:
    return [{"reference_key": ref, "mobile_key": mob} for ref, mob in pairs]


def _pairs_from_dict(items: Sequence[dict]) -> list[Pair]:
    return [(int(item["reference_key"]), int(item["mobile_key"])) for item in items]
