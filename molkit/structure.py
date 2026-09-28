"""分子结构与 XYZ 文件读写。

结构对象只承载数据；任何对齐操作都返回新对象，不修改原件。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def normalize_element(raw: str) -> str:
    """把 'cl'、' CL ' 之类的写法规范成 'Cl'。"""
    token = str(raw).strip()
    if not token:
        raise ValueError("元素符号为空")
    return token[0].upper() + token[1:].lower()


@dataclass(eq=False)
class Atom:
    """单个原子。

    serial 为输入文件中的编号（XYZ 里通常不存在，此时为 None）；
    index 是原子在结构中的从 0 开始的位置。
    """

    element: str
    xyz: np.ndarray
    serial: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        self.element = normalize_element(self.element)
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)

    def copy_with(self, xyz: np.ndarray) -> "Atom":
        return replace(self, xyz=np.asarray(xyz, dtype=float).reshape(3).copy())


@dataclass
class Structure:
    atoms: list[Atom] = field(default_factory=list)
    comment: str = ""

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def coords(self) -> np.ndarray:
        """返回 (n, 3) 坐标数组的副本，外部修改不会影响结构。"""
        if not self.atoms:
            return np.empty((0, 3), dtype=float)
        return np.array([atom.xyz for atom in self.atoms], dtype=float)

    @property
    def serials(self) -> list[int | None]:
        return [atom.serial for atom in self.atoms]

    def duplicate_serials(self) -> list[int]:
        seen: set[int] = set()
        dups: set[int] = set()
        for atom in self.atoms:
            if atom.serial is None:
                continue
            if atom.serial in seen:
                dups.add(atom.serial)
            seen.add(atom.serial)
        return sorted(dups)


def read_xyz(path: str | Path, *, strict: bool = True) -> Structure:
    """读取标准 XYZ 文件。

    第一行原子数，第二行注释，其后每行 ``元素 x y z``；
    允许行尾附加原子名/标签等额外字段（取第四列作为 name）。
    strict=True 时，原子行数与声明数目不符会报错。
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"空文件: {path}")
    try:
        declared = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"首行应为原子数: {lines[0]!r}") from exc

    comment = lines[1] if len(lines) > 1 else ""
    atoms: list[Atom] = []
    for offset, raw in enumerate(lines[2:]):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            if strict:
                raise ValueError(
                    f"第 {offset + 3} 行字段不足，应为 元素 x y z: {raw!r}"
                )
            continue
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise ValueError(f"第 {offset + 3} 行坐标无法解析: {raw!r}") from exc
        name = parts[4] if len(parts) > 4 else None
        atoms.append(Atom(element=parts[0], xyz=xyz, name=name))

    if strict and len(atoms) != declared:
        raise ValueError(
            f"声明 {declared} 个原子，实际解析出 {len(atoms)} 个（可能存在缺失行）"
        )
    return Structure(atoms=atoms, comment=comment)


def structure_from_coords(
    elements: Sequence[str],
    coords: Sequence[float] | np.ndarray,
    *,
    serials: Iterable[int | None] | None = None,
    names: Iterable[str | None] | None = None,
    comment: str = "",
) -> Structure:
    """便于程序化构造结构（测试或其他读取器复用）。"""

    coords_arr = np.asarray(coords, dtype=float).reshape(-1, 3)
    serial_list = list(serials) if serials is not None else [None] * len(elements)
    name_list = list(names) if names is not None else [None] * len(elements)
    if not (len(elements) == len(coords_arr) == len(serial_list) == len(name_list)):
        raise ValueError("元素、坐标、编号、名称数量不一致")
    atoms = [
        Atom(element=element, xyz=row, serial=serial, name=name)
        for element, row, serial, name in zip(
            elements, coords_arr, serial_list, name_list
        )
    ]
    return Structure(atoms=atoms, comment=comment)


def write_xyz(path: str | Path, structure: Structure, *, comment: str | None = None) -> None:
    """写出 XYZ；不修改传入的结构对象。"""

    lines = [str(len(structure)), comment if comment is not None else structure.comment]
    for atom in structure.atoms:
        x, y, z = atom.xyz
        suffix = f"  {atom.name}" if atom.name else ""
        lines.append(f"{atom.element:<4s} {x: .10f}  {y: .10f}  {z: .10f}{suffix}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
