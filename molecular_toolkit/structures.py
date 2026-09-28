from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(eq=False)
class Atom:
    key: int
    element: str
    xyz: np.ndarray
    serial: int | str | None = None
    name: str | None = None
    residue_name: str | None = None
    residue_seq: int | str | None = None
    chain: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.xyz, dtype=float)
        if array.shape != (3,):
            raise ValueError("Atom.xyz must contain three coordinates")
        self.xyz = array.copy()
        self.element = self.element.strip().capitalize()

    @property
    def label(self) -> str:
        if self.serial is None:
            return f"#{self.key}"
        return str(self.serial)


@dataclass(eq=False)
class Structure:
    atoms: list[Atom]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if [atom.key for atom in self.atoms] != list(range(len(self.atoms))):
            raise ValueError("Atom keys must be dense zero-based occurrence keys")
        self.metadata = dict(self.metadata)
        self.atoms = [replace(atom, xyz=np.array(atom.xyz, dtype=float, copy=True)) for atom in self.atoms]

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def coordinates(self) -> np.ndarray:
        if not self.atoms:
            return np.empty((0, 3), dtype=float)
        return np.vstack([atom.xyz for atom in self.atoms])

    def atom(self, key: int) -> Atom:
        if not isinstance(key, int) or key < 0 or key >= len(self.atoms):
            raise KeyError(f"Unknown atom occurrence key: {key!r}")
        return self.atoms[key]

    def copy(self) -> "Structure":
        return Structure(
            [replace(atom, xyz=atom.xyz.copy()) for atom in self.atoms],
            copy_metadata(self.metadata),
        )


def copy_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_structure(path: str | Path) -> Structure:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return load_xyz(path)
    if suffix == ".json":
        return load_json(path)
    if suffix in {".pdb", ".ent"}:
        return load_pdb(path)
    raise ValueError(f"Unsupported structure format: {suffix}")


def save_structure(structure: Structure, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        save_xyz(structure, path)
    elif suffix == ".json":
        save_json(structure, path)
    elif suffix in {".pdb", ".ent"}:
        save_pdb(structure, path)
    else:
        raise ValueError(f"Unsupported structure format: {suffix}")


def structure_from_dict(data: dict[str, Any]) -> Structure:
    atoms = []
    for key, item in enumerate(data.get("atoms", [])):
        xyz = item.get("xyz", item.get("coordinates"))
        atoms.append(
            Atom(
                key=key,
                element=str(item["element"]),
                xyz=xyz,
                serial=item.get("serial", item.get("id")),
                name=item.get("name"),
                residue_name=item.get("residue_name"),
                residue_seq=item.get("residue_seq"),
                chain=item.get("chain"),
            )
        )
    metadata = copy_metadata(data.get("metadata", {}))
    return Structure(atoms, metadata)


def structure_to_dict(structure: Structure) -> dict[str, Any]:
    return {
        "metadata": copy_metadata(structure.metadata),
        "atoms": [
            {
                "key": atom.key,
                "serial": atom.serial,
                "element": atom.element,
                "xyz": atom.xyz.tolist(),
                "name": atom.name,
                "residue_name": atom.residue_name,
                "residue_seq": atom.residue_seq,
                "chain": atom.chain,
            }
            for atom in structure.atoms
        ],
    }


def load_json(path: str | Path) -> Structure:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return structure_from_dict(data)


def save_json(structure: Structure, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(structure_to_dict(structure), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_xyz(path: str | Path) -> Structure:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Empty XYZ file")
    count = int(lines[0].strip())
    atoms: list[Atom] = []
    for line_number, line in enumerate(lines[2 : 2 + count], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line {line_number}")
        atoms.append(Atom(len(atoms), fields[0], [float(x) for x in fields[1:4]]))
    if len(atoms) != count:
        raise ValueError(f"XYZ declares {count} atoms but contains {len(atoms)}")
    return Structure(atoms, {"comment": lines[1].strip() if len(lines) > 1 else ""})


def save_xyz(structure: Structure, path: str | Path) -> None:
    lines = [str(len(structure)), structure.metadata.get("comment", "")]
    for atom in structure.atoms:
        x, y, z = atom.xyz
        lines.append(f"{atom.element:<2s} {x: .8f} {y: .8f} {z: .8f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pdb(path: str | Path) -> Structure:
    atoms: list[Atom] = []
    title: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = line[0:6].strip()
        if record == "TITLE":
            title.append(line[10:80].strip())
        if record not in {"ATOM", "HETATM"}:
            continue
        serial_text = line[6:11].strip()
        serial: int | str | None
        try:
            serial = int(serial_text) if serial_text else None
        except ValueError:
            serial = serial_text
        element = line[76:78].strip() or line[12:16].strip()[0]
        atoms.append(
            Atom(
                key=len(atoms),
                element=element,
                xyz=[float(line[30:38]), float(line[38:46]), float(line[46:54])],
                serial=serial,
                name=line[12:16].strip(),
                residue_name=line[17:20].strip(),
                chain=line[21:22].strip() or None,
                residue_seq=parse_optional_int(line[22:26]),
            )
        )
    return Structure(atoms, {"title": " ".join(title)})


def parse_optional_int(text: str) -> int | None:
    text = text.strip()
    return int(text) if text else None


def save_pdb(structure: Structure, path: str | Path) -> None:
    lines = []
    title = str(structure.metadata.get("title", ""))
    if title:
        lines.append(f"TITLE     {title[:70]}")
    for atom in structure.atoms:
        serial = atom.key + 1 if atom.serial is None else atom.serial
        serial_text = str(serial)[-5:]
        name = (atom.name or atom.element).ljust(4)[:4]
        residue = (atom.residue_name or "UNK").rjust(3)[:3]
        chain = (atom.chain or " ")[:1]
        residue_seq = atom.residue_seq if atom.residue_seq is not None else 1
        x, y, z = atom.xyz
        element = atom.element.rjust(2)[:2]
        lines.append(
            f"ATOM  {serial_text:>5} {name} {residue} {chain}{int(residue_seq):4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}"
        )
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def transform_structure(
    structure: Structure, rotation: np.ndarray, translation: Iterable[float] | np.ndarray
) -> Structure:
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected a 3x3 rotation and a length-3 translation")
    moved = structure.copy()
    for atom in moved.atoms:
        atom.xyz = rotation @ atom.xyz + translation
    moved.metadata["transform"] = {
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }
    return moved
