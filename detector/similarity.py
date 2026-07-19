from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from config import Config
from detector.features import BoundingBox, FeatureVector
from detector.graph import GraphMetrics
from detector.reader import CadDocument
from detector.sequence import SequenceComparisonResult


_JUSTIFICATION_LABELS: Dict[str, str] = {
    "geometry": "Geometria",
    "sequence": "Sequência",
    "graph": "Grafo",
    "styles": "Estilos/Layers",
}


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float
    weight: float
    justification: str


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


def _named_set_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


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


def _optional_equality_similarity(a: Optional[str], b: Optional[str]) -> float:
    if a is None and b is None:
        return 0.5
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


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

    def compare_styles(self, document_a: CadDocument, document_b: CadDocument) -> float:
        components = (
            _named_set_jaccard(document_a.layers, document_b.layers),
            _named_set_jaccard(document_a.text_styles, document_b.text_styles),
            _named_set_jaccard(document_a.dimension_styles, document_b.dimension_styles),
            _named_set_jaccard(document_a.blocks, document_b.blocks),
        )
        return _clamp01(sum(components) / len(components))

    def compute_score(
        self,
        feature_a: FeatureVector,
        feature_b: FeatureVector,
        sequence_result: SequenceComparisonResult,
        graph_a: GraphMetrics,
        graph_b: GraphMetrics,
        document_a: CadDocument,
        document_b: CadDocument,
    ) -> SimilarityResult:
        component_scores = (
            self._component("geometry", self.compare_geometry(feature_a, feature_b)),
            self._component("sequence", self.compare_sequence(sequence_result)),
            self._component("graph", self.compare_graph(graph_a, graph_b)),
            self._component("styles", self.compare_styles(document_a, document_b)),
        )
        total = sum(component.score * component.weight for component in component_scores)
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

    def _component(self, name: str, score: float) -> ComponentScore:
        weight = self._config.weights.get(name, 0.0)
        return ComponentScore(
            name=name,
            score=score,
            weight=weight,
            justification=self._justify(name, score),
        )

    def _justify(self, name: str, score: float) -> str:
        label = _JUSTIFICATION_LABELS.get(name, name)
        percentage = round(score * 100.0, 1)
        return f"{label}: {percentage}% de similaridade"
