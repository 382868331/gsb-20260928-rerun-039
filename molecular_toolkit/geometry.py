from __future__ import annotations

import numpy as np


class GeometryError(ValueError):
    pass


class ReflectionAlignmentError(GeometryError):
    pass


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, degrees: bool = True) -> float:
    vector_ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    vector_bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    value = np.arccos(_unit_dot(vector_ba, vector_bc))
    return float(np.degrees(value) if degrees else value)


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, degrees: bool = True) -> float:
    p0, p1, p2, p3 = [np.asarray(point, dtype=float) for point in (a, b, c, d)]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m1 = np.cross(_normalize(b1), n1)
    x = float(np.dot(_normalize(n1), _normalize(n2)))
    y = float(np.dot(m1, _normalize(n2)))
    value = np.arctan2(y, x)
    return float(np.degrees(value) if degrees else value)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("RMSD requires two Nx3 coordinate arrays")
    if len(a) == 0:
        raise ValueError("RMSD requires at least one point")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def kabsch(
    reference: np.ndarray,
    mobile: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    reference = np.asarray(reference, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Kabsch requires two matching Nx3 coordinate arrays")
    if len(reference) < 3:
        raise GeometryError("At least three matched atoms are required for rigid alignment")

    reference_center = reference.mean(axis=0)
    mobile_center = mobile.mean(axis=0)
    reference_centered = reference - reference_center
    mobile_centered = mobile - mobile_center
    if np.linalg.matrix_rank(np.vstack([mobile_centered, reference_centered]), tol=1e-8) < 3:
        raise GeometryError("Matched atoms must span three-dimensional space")

    covariance = mobile_centered.T @ reference_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    if singular_values[1] < 1e-10:
        raise GeometryError("Matched atoms are collinear or nearly collinear")

    reflected_rotation = vt.T @ u.T
    is_reflection = float(np.linalg.det(reflected_rotation)) < 0
    if is_reflection and not allow_reflection:
        raise ReflectionAlignmentError(
            "Best orthogonal alignment is a reflection; atom ordering or enantiomer mapping may be wrong. "
            "Pass allow_reflection=True only when an improper transformation is intended."
        )
    rotation = reflected_rotation
    if is_reflection:
        correction = np.diag([1.0, 1.0, -1.0])
        rotation = vt.T @ correction @ u.T
    translation = reference_center - rotation @ mobile_center
    aligned = mobile @ rotation.T + translation
    matched_rmsd = rmsd(reference, aligned)
    if allow_reflection and is_reflection:
        improper_translation = reference_center - reflected_rotation @ mobile_center
        improper_aligned = mobile @ reflected_rotation.T + improper_translation
        return reflected_rotation, improper_translation, rmsd(reference, improper_aligned), True
    return rotation, translation, matched_rmsd, is_reflection


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise GeometryError("Zero-length vector in geometric calculation")
    return vector / norm


def _unit_dot(a: np.ndarray, b: np.ndarray) -> float:
    value = float(np.dot(_normalize(a), _normalize(b)))
    return min(1.0, max(-1.0, value))
