from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector.features import _ENTITY_TYPE_NAMES, _representative_points
from detector.reader import CadDocument, CadEntity, Point2D


_EPSILON = 1e-9
_SIGN_FLIPS: Tuple[Tuple[float, float], ...] = (
    (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
)


@dataclass(frozen=True)
class PieceFeatures:
    entity_count: int
    entity_type_counts: Tuple[Tuple[str, int], ...]
    pca_points: Tuple[Tuple[float, float], ...]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio_similarity(a: float, b: float) -> float:
    if a <= 0.0 and b <= 0.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return _clamp01(min(a, b) / max(a, b))


def _type_count_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(key, 0), b.get(key, 0)) for key in keys)
    denominator = sum(max(a.get(key, 0), b.get(key, 0)) for key in keys)
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _min_point_distance(points_a: Sequence[Point2D], points_b: Sequence[Point2D]) -> float:
    return min(
        math.hypot(ax - bx, ay - by) for ax, ay in points_a for bx, by in points_b
    )


def _build_adjacency_graph(
    entity_points: Sequence[Sequence[Point2D]], tolerance: float
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(entity_points)))
    for i in range(len(entity_points)):
        if not entity_points[i]:
            continue
        for j in range(i + 1, len(entity_points)):
            if not entity_points[j]:
                continue
            if _min_point_distance(entity_points[i], entity_points[j]) <= tolerance:
                graph.add_edge(i, j)
    return graph


def _pca_align(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = math.sqrt(float((centered ** 2).sum(axis=1).mean()))
    if scale < _EPSILON:
        return centered
    normalized = centered / scale
    covariance = np.cov(normalized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    return normalized @ basis


def _chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return math.inf
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0)


def _best_chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = math.inf
    for sign_x, sign_y in _SIGN_FLIPS:
        flipped = points_b * np.array([sign_x, sign_y])
        best = min(best, _chamfer_distance(points_a, flipped))
    return best


def _shape_similarity(piece_a: PieceFeatures, piece_b: PieceFeatures) -> float:
    if len(piece_a.pca_points) < 2 and len(piece_b.pca_points) < 2:
        return 1.0
    if len(piece_a.pca_points) < 2 or len(piece_b.pca_points) < 2:
        return 0.0
    distance = _best_chamfer_distance(
        np.array(piece_a.pca_points), np.array(piece_b.pca_points)
    )
    return _clamp01(math.exp(-distance))


def _piece_similarity(piece_a: PieceFeatures, piece_b: PieceFeatures) -> float:
    type_score = _type_count_similarity(
        dict(piece_a.entity_type_counts), dict(piece_b.entity_type_counts)
    )
    point_count_score = _ratio_similarity(
        len(piece_a.pca_points), len(piece_b.pca_points)
    )
    shape_score = _shape_similarity(piece_a, piece_b)
    return (type_score + point_count_score + shape_score) / 3.0


def _build_piece(entities: Sequence[CadEntity]) -> PieceFeatures:
    points: List[Point2D] = []
    for entity in entities:
        points.extend(_representative_points(entity))
    type_counts: Dict[str, int] = {}
    for entity in entities:
        name = _ENTITY_TYPE_NAMES.get(type(entity))
        if name is not None:
            type_counts[name] = type_counts.get(name, 0) + 1
    if len(points) >= 2:
        pca_points = tuple(
            map(tuple, _pca_align(np.array(points, dtype=float)))
        )
    else:
        pca_points = ()
    return PieceFeatures(
        entity_count=len(entities),
        entity_type_counts=tuple(sorted(type_counts.items())),
        pca_points=pca_points,
    )


def extract_pieces(document: CadDocument, tolerance: float) -> Tuple[PieceFeatures, ...]:
    entities = document.entities
    entity_points = [_representative_points(entity) for entity in entities]
    graph = _build_adjacency_graph(entity_points, tolerance)
    pieces: List[PieceFeatures] = []
    for component_nodes in nx.connected_components(graph):
        indices = sorted(component_nodes)
        pieces.append(_build_piece([entities[index] for index in indices]))
    return tuple(pieces)


def compare_piece_sets(
    pieces_a: Sequence[PieceFeatures], pieces_b: Sequence[PieceFeatures]
) -> Optional[float]:
    if not pieces_a and not pieces_b:
        return None
    if not pieces_a or not pieces_b:
        return 0.0
    similarity_matrix = np.array(
        [[_piece_similarity(piece_a, piece_b) for piece_b in pieces_b] for piece_a in pieces_a]
    )
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    matched_scores = similarity_matrix[row_indices, col_indices]
    coverage = min(len(pieces_a), len(pieces_b)) / max(len(pieces_a), len(pieces_b))
    return _clamp01(float(matched_scores.mean()) * coverage)
