from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .candidates import CorrespondenceCandidate
from .correspondence import AtomCorrespondence, Pair
from .regions import RegionAlignmentResult, align_on_stable_atoms
from .structures import Structure, load_structure, save_structure


@dataclass(eq=False)
class Conformer:
    name: str
    structure: Structure
    source_path: str | None = None
    source_format: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> "Conformer":
        path = Path(path)
        structure = load_structure(path)
        return cls(
            name=name or structure.metadata.get("provenance", {}).get("structure_name", path.stem),
            structure=structure,
            source_path=str(path),
            source_format=path.suffix.lstrip("."),
        )

    def provenance(self) -> dict:
        provenance = dict(self.structure.metadata.get("provenance", {}))
        provenance.update(
            {
                "name": self.name,
                "source_path": self.source_path or provenance.get("source_path"),
                "source_format": self.source_format or provenance.get("source_format"),
                "atom_count": len(self.structure),
            }
        )
        return provenance


@dataclass(eq=False)
class ConformerComparison:
    conformer: Conformer
    correspondence: AtomCorrespondence
    alignment: RegionAlignmentResult
    candidate_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "conformer": self.conformer.provenance(),
            "candidate_id": self.candidate_id,
            "correspondence": correspondence_to_dict(self.correspondence),
            "alignment": self.alignment.to_dict(),
        }


def compare_conformer(
    reference: Structure,
    conformer: Conformer,
    correspondence: AtomCorrespondence | CorrespondenceCandidate,
    stable_reference_keys: Iterable[int],
    regions: dict[str, tuple[Iterable[int], Iterable[int] | None]] | None = None,
) -> ConformerComparison:
    if isinstance(correspondence, CorrespondenceCandidate):
        candidate_id = correspondence.candidate_id
        atom_correspondence = correspondence.to_correspondence()
    else:
        candidate_id = None
        atom_correspondence = correspondence
    alignment = align_on_stable_atoms(
        reference,
        conformer.structure,
        atom_correspondence.pairs,
        stable_reference_keys,
        regions,
    )
    return ConformerComparison(
        conformer=conformer,
        correspondence=atom_correspondence,
        alignment=alignment,
        candidate_id=candidate_id,
    )


def compare_ensemble(
    reference: Structure | Conformer,
    conformers: Sequence[tuple[Conformer, AtomCorrespondence | CorrespondenceCandidate]],
    stable_reference_keys: Iterable[int],
    regions: dict[str, tuple[Iterable[int], Iterable[int] | None]] | None = None,
) -> list[ConformerComparison]:
    if isinstance(reference, Conformer):
        reference = reference.structure
    return [
        compare_conformer(reference, conformer, correspondence, stable_reference_keys, regions)
        for conformer, correspondence in conformers
    ]


def save_comparison_report(
    comparisons: Sequence[ConformerComparison],
    report_path: str | Path,
    *,
    reference: Structure | Conformer | None = None,
    aligned_directory: str | Path | None = None,
    notes: Sequence[str] | None = None,
) -> None:
    """Save a traceable JSON report and optionally aligned conformer structures.

    Original structures are never rewritten. Aligned outputs contain transform
    metadata, and the JSON records source path/format, candidate id and exact
    atom pairs used for every conformer.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_dir = Path(aligned_directory) if aligned_directory is not None else None
    if aligned_dir is not None:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for comparison in comparisons:
        record = comparison.to_dict()
        if aligned_dir is not None:
            suffix = f".{comparison.conformer.source_format or 'xyz'}"
            aligned_path = aligned_dir / f"{comparison.conformer.name}_aligned{suffix}"
            save_structure(comparison.alignment.aligned_mobile, aligned_path)
            record["aligned_structure"] = str(aligned_path)
        records.append(record)

    reference_info = None
    if isinstance(reference, Conformer):
        reference_info = reference.provenance()
    elif isinstance(reference, Structure):
        reference_info = {
            "provenance": reference.metadata.get("provenance"),
            "atom_count": len(reference),
        }

    report = {
        "reference": reference_info,
        "conformer_count": len(comparisons),
        "conformers": records,
        "notes": list(notes or []),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def correspondence_to_dict(correspondence: AtomCorrespondence) -> dict:
    return {
        "pairs": [
            {"reference_key": ref_key, "mobile_key": mobile_key}
            for ref_key, mobile_key in correspondence.pairs
        ],
        "unmatched_reference": correspondence.unmatched_reference,
        "unmatched_mobile": correspondence.unmatched_mobile,
        "ambiguous_identities": correspondence.ambiguous_identities,
        "notes": correspondence.notes,
    }


class ComparisonSequence:
    """Ordered, conformer-order view over ensemble comparisons.

    Comparisons are kept in construction order (the conformer order) rather
    than sorted by RMSD or candidate score, so results can be stepped through
    in trajectory order. Indexing and ``step`` accept negative indices.
    """

    def __init__(self, comparisons: Sequence[ConformerComparison]):
        self._comparisons = list(comparisons)

    def __len__(self) -> int:
        return len(self._comparisons)

    def __iter__(self):
        return iter(self._comparisons)

    def __getitem__(self, index: int | slice) -> ConformerComparison | list[ConformerComparison]:
        return self._comparisons[index]

    @property
    def names(self) -> list[str]:
        return [comparison.conformer.name for comparison in self._comparisons]

    def get(self, index: int) -> ConformerComparison:
        return self._comparisons[index]

    def step(self, index: int, delta: int = 1) -> ConformerComparison:
        if not self._comparisons:
            raise IndexError("Empty comparison sequence")
        return self._comparisons[(index + delta) % len(self._comparisons)]

    def transitions(self) -> list[dict]:
        """Region metric changes between successive conformers in order."""
        changes = []
        for current, following in zip(self._comparisons, self._comparisons[1:]):
            region_names = set(current.alignment.regions) | set(following.alignment.regions)
            region_change = {}
            for name in sorted(region_names):
                before = current.alignment.regions.get(name)
                after = following.alignment.regions.get(name)
                region_change[name] = {
                    "rmsd_before": None if before is None else before.rmsd,
                    "rmsd_after": None if after is None else after.rmsd,
                    "rmsd_delta": (
                        None
                        if before is None or after is None
                        else after.rmsd - before.rmsd
                    ),
                }
            changes.append(
                {
                    "from": current.conformer.name,
                    "to": following.conformer.name,
                    "regions": region_change,
                }
            )
        return changes
