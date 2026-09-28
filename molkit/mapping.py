"""原子身份对应。

原则：对应关系不能依赖文件中的出现顺序。

* 调用者可用 (ref_index, mob_index) 给出显式对应（也可用原子编号解析得到）；
* 未显式指定的原子只在某元素两侧各剩一个时自动配对；
* 对称等价（同元素多个）或两侧数量不一致时，不做猜测，
  原样列入 ambiguous_groups / 未匹配列表，交调用者决定；
* 重复编号直接报错，因为编号不再能唯一标识原子。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .structure import Structure


@dataclass
class AmbiguousGroup:
    element: str
    ref_indices: list[int]
    mob_indices: list[int]
    reason: str


@dataclass
class MappingResult:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    unmatched_ref: list[int] = field(default_factory=list)
    unmatched_mob: list[int] = field(default_factory=list)
    ambiguous_groups: list[AmbiguousGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """两侧所有原子都已一一对应。"""
        return not self.unmatched_ref and not self.unmatched_mob


def _validate_explicit_pairs(
    ref: Structure,
    mob: Structure,
    pairs: list[tuple[int, int]],
    strict_elements: bool,
) -> None:
    seen_ref: set[int] = set()
    seen_mob: set[int] = set()
    for ref_i, mob_i in pairs:
        if not isinstance(ref_i, int) or not isinstance(mob_i, int):
            raise TypeError(f"对应索引必须是 int: {(ref_i, mob_i)!r}")
        if not 0 <= ref_i < len(ref):
            raise IndexError(f"参考结构索引越界: {ref_i}（共 {len(ref)} 个原子）")
        if not 0 <= mob_i < len(mob):
            raise IndexError(f"待对齐结构索引越界: {mob_i}（共 {len(mob)} 个原子）")
        if ref_i in seen_ref:
            raise ValueError(f"参考结构原子 {ref_i} 被重复对应")
        if mob_i in seen_mob:
            raise ValueError(f"待对齐结构原子 {mob_i} 被重复对应")
        seen_ref.add(ref_i)
        seen_mob.add(mob_i)
        if strict_elements and ref.atoms[ref_i].element != mob.atoms[mob_i].element:
            raise ValueError(
                f"对应 ({ref_i}, {mob_i}) 元素不一致: "
                f"{ref.atoms[ref_i].element} vs {mob.atoms[mob_i].element}"
            )


def build_mapping(
    ref: Structure,
    mob: Structure,
    explicit_pairs: list[tuple[int, int]] | None = None,
    *,
    strict_elements: bool = True,
) -> MappingResult:
    """构造原子对应。

    explicit_pairs 中的每个元素为 (ref_index, mob_index)。
    剩余原子按元素分组：两侧都只剩一个时自动配对；
    同元素有多个（对称位置）或数量不一致时不猜测，
    进入 ambiguous_groups 与未匹配列表。
    """

    pairs: list[tuple[int, int]] = [tuple(p) for p in (explicit_pairs or [])]
    _validate_explicit_pairs(ref, mob, pairs, strict_elements)

    matched_ref = {p[0] for p in pairs}
    matched_mob = {p[1] for p in pairs}

    ref_groups: dict[str, list[int]] = {}
    mob_groups: dict[str, list[int]] = {}
    for i, atom in enumerate(ref.atoms):
        if i not in matched_ref:
            ref_groups.setdefault(atom.element, []).append(i)
    for i, atom in enumerate(mob.atoms):
        if i not in matched_mob:
            mob_groups.setdefault(atom.element, []).append(i)

    result = MappingResult(pairs=sorted(pairs))
    for element in sorted(set(ref_groups) | set(mob_groups)):
        refs = ref_groups.get(element, [])
        mobs = mob_groups.get(element, [])
        if len(refs) == 1 and len(mobs) == 1:
            result.pairs.append((refs[0], mobs[0]))
            continue
        if refs and not mobs:
            result.unmatched_ref.extend(refs)
            result.warnings.append(f"元素 {element}: 参考结构多出 {len(refs)} 个原子 {refs}")
            continue
        if mobs and not refs:
            result.unmatched_mob.extend(mobs)
            result.warnings.append(f"元素 {element}: 待对齐结构多出 {len(mobs)} 个原子 {mobs}")
            continue
        if len(refs) == len(mobs):
            reason = (
                f"元素 {element} 两侧各有 {len(refs)} 个对称等价原子，"
                "无法仅凭输入顺序确定身份，请提供显式对应"
            )
        else:
            reason = (
                f"元素 {element} 两侧数量不一致（参考 {len(refs)} 个、"
                f"待对齐 {len(mobs)} 个），无法判断哪些缺失，请提供显式对应"
            )
        result.ambiguous_groups.append(
            AmbiguousGroup(element=element, ref_indices=refs, mob_indices=mobs, reason=reason)
        )
        result.unmatched_ref.extend(refs)
        result.unmatched_mob.extend(mobs)

    result.pairs.sort()
    result.unmatched_ref.sort()
    result.unmatched_mob.sort()
    return result


def resolve_serial_pairs(
    ref: Structure,
    mob: Structure,
) -> list[tuple[int, int]]:
    """按原子编号建立索引对。

    任一结构出现重复编号都会抛 ValueError；编号缺失（None）的原子被忽略。
    """

    for label, structure in (("参考结构", ref), ("待对齐结构", mob)):
        dups = structure.duplicate_serials()
        if dups:
            raise ValueError(f"{label}存在重复原子编号: {dups}，无法据此确定身份")

    ref_map = {
        atom.serial: i
        for i, atom in enumerate(ref.atoms)
        if atom.serial is not None
    }
    pairs: list[tuple[int, int]] = []
    for mob_i, atom in enumerate(mob.atoms):
        if atom.serial is not None and atom.serial in ref_map:
            pairs.append((ref_map[atom.serial], mob_i))
    return pairs
