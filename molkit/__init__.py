"""分子结构研究工具包：结构读取、原子身份对应与几何对齐。"""

from .structure import Atom, Structure, read_xyz, structure_from_coords, write_xyz
from .mapping import MappingResult, build_mapping, resolve_serial_pairs
from .geometry import (
    AlignmentResult,
    align_structures,
    apply_transform,
    kabsch,
    rmsd,
    save_alignment,
    save_pair_report,
)

__all__ = [
    "Atom",
    "Structure",
    "read_xyz",
    "structure_from_coords",
    "write_xyz",
    "MappingResult",
    "build_mapping",
    "resolve_serial_pairs",
    "AlignmentResult",
    "align_structures",
    "apply_transform",
    "kabsch",
    "rmsd",
    "save_alignment",
    "save_pair_report",
]
