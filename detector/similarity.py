from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rapidfuzz.distance import Levenshtein

from config import Config
from detector.features import BoundingBox, FeatureVector
from detector.graph import GraphMetrics
from detector.pieces import compare_piece_sets, extract_pieces
from detector.reader import CadDocument
from detector.sequence import SequenceComparisonResult, TextSequenceComparisonResult


_JUSTIFICATION_LABELS: Dict[str, str] = {
    "geometry": "Geometria",
    "sequence": "Sequência",
    "graph": "Grafo",
    "dimensions": "Cotas",
    "text": "Texto",
    "blocks": "Blocos",
    "styles": "Estilos",
    "decimal_precision": "Precisão Decimal",
    "pieces": "Peças (casamento geométrico)",
}

_EPSILON = 1e-9


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float
    weight: float
    justification: str
    applicable: bool = True


@dataclass(frozen=True)
class SimilarityResult:
    total_score: float
    component_scores: Tuple[ComponentScore, ...]

    def component(self, name: str) -> ComponentScore:
        for component in self.component_scores:
            if component.name == name:
                return component
        raise KeyError(name)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _histogram_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return _clamp01(sum(min(x, y) for x, y in zip(a, b)))


def _is_zero_histogram(histogram: Sequence[float]) -> bool:
    return not any(value > _EPSILON for value in histogram)


def _counts_to_dict(counts: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    return dict(counts)


def _count_dict_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(key, 0), b.get(key, 0)) for key in keys)
    denominator = sum(max(a.get(key, 0), b.get(key, 0)) for key in keys)
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _bounding_box_similarity(box_a: BoundingBox, box_b: BoundingBox) -> float:
    area_a, area_b = box_a.area, box_b.area
    if area_a <= 0.0 and area_b <= 0.0:
        return 1.0
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return _clamp01(min(area_a, area_b) / max(area_a, area_b))


def _ratio_similarity(a: float, b: float) -> float:
    if a <= 0.0 and b <= 0.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return _clamp01(min(a, b) / max(a, b))


def _mean_value(values: Sequence[Tuple[int, float]]) -> float:
    if not values:
        return 0.0
    return sum(value for _, value in values) / len(values)


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return _clamp01(dot / (norm_a * norm_b))


def _optional_equality_similarity(a: Optional[str], b: Optional[str]) -> float:
    if a is None and b is None:
        return 0.5
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def _text_content_similarity(
    texts_a: Sequence[Tuple[str, float, float]],
    texts_b: Sequence[Tuple[str, float, float]],
) -> float:
    if not texts_a and not texts_b:
        return 1.0
    if not texts_a or not texts_b:
        return 0.0
    strings_a = [t[0] for t in texts_a]
    strings_b = [t[0] for t in texts_b]
    total = 0.0
    for a in strings_a:
        best = max(
            Levenshtein.normalized_similarity(a, b) for b in strings_b
        )
        total += best
    return _clamp01(total / len(strings_a))


def _text_position_similarity(
    texts_a: Sequence[Tuple[str, float, float]],
    texts_b: Sequence[Tuple[str, float, float]],
) -> float:
    if not texts_a and not texts_b:
        return 1.0
    if not texts_a or not texts_b:
        return 0.0
    all_points_a = [(t[1], t[2]) for t in texts_a]
    all_points_b = [(t[1], t[2]) for t in texts_b]
    xs_a = [p[0] for p in all_points_a]
    ys_a = [p[1] for p in all_points_a]
    xs_b = [p[0] for p in all_points_b]
    ys_b = [p[1] for p in all_points_b]
    if not xs_a or not xs_b:
        return 0.0
    max_distance = max(
        max(xs_a) - min(xs_a), max(ys_a) - min(ys_a),
        max(xs_b) - min(xs_b), max(ys_b) - min(ys_b),
    )
    if max_distance <= _EPSILON:
        max_distance = 1.0
    total = 0.0
    for pa in all_points_a:
        nearest = min(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pb in all_points_b)
        total += max(0.0, 1.0 - nearest / max_distance)
    return _clamp01(total / len(all_points_a))


class SimilarityEngine:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config if config is not None else Config()

    def compare_geometry(self, feature_a: FeatureVector, feature_b: FeatureVector) -> float:
        components = (
            _count_dict_similarity(
                _counts_to_dict(feature_a.entity_type_counts),
                _counts_to_dict(feature_b.entity_type_counts),
            ),
            _histogram_similarity(feature_a.angle_histogram, feature_b.angle_histogram),
            _histogram_similarity(feature_a.length_histogram, feature_b.length_histogram),
            _bounding_box_similarity(feature_a.bounding_box, feature_b.bounding_box),
            _histogram_similarity(feature_a.spatial_density, feature_b.spatial_density),
        )
        return _clamp01(sum(components) / len(components))

    def compare_sequence(self, sequence_result: SequenceComparisonResult) -> float:
        return _clamp01(sequence_result.combined_score)

    def compare_graph(self, graph_a: GraphMetrics, graph_b: GraphMetrics) -> float:
        components = (
            _ratio_similarity(self._edge_density(graph_a), self._edge_density(graph_b)),
            _ratio_similarity(float(graph_a.component_count), float(graph_b.component_count)),
            _ratio_similarity(
                _mean_value(graph_a.degree_centrality), _mean_value(graph_b.degree_centrality)
            ),
            _ratio_similarity(
                _mean_value(graph_a.clustering_coefficient),
                _mean_value(graph_b.clustering_coefficient),
            ),
            _ratio_similarity(
                _mean_value(graph_a.betweenness_centrality),
                _mean_value(graph_b.betweenness_centrality),
            ),
        )
        return _clamp01(sum(components) / len(components))

    def compare_dimensions(
        self, feature_a: FeatureVector, feature_b: FeatureVector
    ) -> Optional[float]:
        hist_a = feature_a.dimension_distance_histogram
        hist_b = feature_b.dimension_distance_histogram
        if _is_zero_histogram(hist_a) and _is_zero_histogram(hist_b):
            return None
        return _histogram_similarity(hist_a, hist_b)

    def compare_text(
        self,
        text_sequence_result: TextSequenceComparisonResult,
        feature_a: FeatureVector,
        feature_b: FeatureVector,
    ) -> Optional[float]:
        if not feature_a.text_positions and not feature_b.text_positions:
            return None
        sequence_score = text_sequence_result.combined_score
        content_score = _text_content_similarity(
            feature_a.text_positions, feature_b.text_positions
        )
        position_score = _text_position_similarity(
            feature_a.text_positions, feature_b.text_positions
        )
        return _clamp01((sequence_score + content_score + position_score) / 3.0)

    def compare_blocks(
        self, feature_a: FeatureVector, feature_b: FeatureVector
    ) -> Optional[float]:
        counts_a = _counts_to_dict(feature_a.block_usage_counts)
        counts_b = _counts_to_dict(feature_b.block_usage_counts)
        keys = set(counts_a) | set(counts_b)
        if not keys:
            return None
        jaccard = _jaccard_similarity(set(counts_a), set(counts_b))
        vector_a = [float(counts_a.get(key, 0)) for key in keys]
        vector_b = [float(counts_b.get(key, 0)) for key in keys]
        cosine = _cosine_similarity(vector_a, vector_b)
        return _clamp01(0.5 * jaccard + 0.5 * cosine)

    def compare_styles(
        self, doc_a: CadDocument, doc_b: CadDocument
    ) -> Optional[float]:
        if (
            not doc_a.text_styles
            and not doc_b.text_styles
            and not doc_a.dimension_styles
            and not doc_b.dimension_styles
        ):
            return None
        text_score = _jaccard_similarity(set(doc_a.text_styles), set(doc_b.text_styles))
        dimension_score = _jaccard_similarity(
            set(doc_a.dimension_styles), set(doc_b.dimension_styles)
        )
        return _clamp01((text_score + dimension_score) / 2.0)

    def compare_decimal_precision(
        self, feature_a: FeatureVector, feature_b: FeatureVector
    ) -> Optional[float]:
        hist_a = feature_a.decimal_precision_histogram
        hist_b = feature_b.decimal_precision_histogram
        if _is_zero_histogram(hist_a) and _is_zero_histogram(hist_b):
            return None
        return _histogram_similarity(hist_a, hist_b)

    def compare_pieces(self, doc_a: CadDocument, doc_b: CadDocument) -> Optional[float]:
        tolerance = self._config.piece_adjacency_tolerance
        pieces_a = extract_pieces(doc_a, tolerance)
        pieces_b = extract_pieces(doc_b, tolerance)
        return compare_piece_sets(pieces_a, pieces_b)

    def compute_score(
        self,
        feature_a: FeatureVector,
        feature_b: FeatureVector,
        sequence_result: SequenceComparisonResult,
        graph_a: GraphMetrics,
        graph_b: GraphMetrics,
        text_sequence_result: TextSequenceComparisonResult,
        doc_a: CadDocument,
        doc_b: CadDocument,
    ) -> SimilarityResult:
        component_scores = (
            self._component("geometry", self.compare_geometry(feature_a, feature_b)),
            self._component("sequence", self.compare_sequence(sequence_result)),
            self._component("graph", self.compare_graph(graph_a, graph_b)),
            self._component("dimensions", self.compare_dimensions(feature_a, feature_b)),
            self._component("text", self.compare_text(text_sequence_result, feature_a, feature_b)),
            self._component("blocks", self.compare_blocks(feature_a, feature_b)),
            self._component("styles", self.compare_styles(doc_a, doc_b)),
            self._component(
                "decimal_precision", self.compare_decimal_precision(feature_a, feature_b)
            ),
            self._component("pieces", self.compare_pieces(doc_a, doc_b)),
        )
        applicable_components = [c for c in component_scores if c.applicable]
        weight_sum = sum(c.weight for c in applicable_components)
        if weight_sum <= 0.0:
            total = 0.0
        else:
            total = sum(
                c.score * c.weight for c in applicable_components
            ) / weight_sum
        return SimilarityResult(
            total_score=_clamp01(total) * 100.0,
            component_scores=component_scores,
        )

    def _edge_density(self, graph: GraphMetrics) -> float:
        if graph.node_count <= 1:
            return 0.0
        max_edges = graph.node_count * (graph.node_count - 1) / 2.0
        if max_edges <= 0.0:
            return 0.0
        return graph.edge_count / max_edges

    def _component(self, name: str, score: Optional[float]) -> ComponentScore:
        weight = self._config.weights.get(name, 0.0)
        applicable = score is not None
        resolved_score = score if applicable else 0.0
        return ComponentScore(
            name=name,
            score=resolved_score,
            weight=weight,
            justification=self._justify(name, resolved_score, applicable),
            applicable=applicable,
        )

    def _justify(self, name: str, score: float, applicable: bool = True) -> str:
        label = _JUSTIFICATION_LABELS.get(name, name)
        if not applicable:
            return f"{label}: sem dados suficientes em ambos os desenhos para comparar"
        percentage = round(score * 100.0, 1)
        return f"{label}: {percentage}% de similaridade"
