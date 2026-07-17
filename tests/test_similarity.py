from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pytest

from config import Config
from detector.features import BoundingBox, FeatureVector
from detector.graph import GraphMetrics
from detector.reader import CadDocument, DocumentMetadata
from detector.sequence import SequenceComparisonResult
from detector.similarity import ComponentScore, SimilarityEngine, SimilarityResult


@pytest.fixture
def engine() -> SimilarityEngine:
    return SimilarityEngine()


def _bounding_box(width: float = 10.0, height: float = 5.0) -> BoundingBox:
    return BoundingBox(
        min_x=0.0, min_y=0.0, max_x=width, max_y=height,
        width=width, height=height, area=width * height,
    )


def _feature_vector(
    entity_type_counts: Tuple[Tuple[str, int], ...] = (("LINE", 4), ("CIRCLE", 2)),
    angle_histogram: Tuple[float, ...] = (0.5, 0.5),
    length_histogram: Tuple[float, ...] = (0.5, 0.5),
    bounding_box: Optional[BoundingBox] = None,
    spatial_density: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    layer_entity_counts: Tuple[Tuple[str, int], ...] = (("0", 6),),
    block_usage_counts: Tuple[Tuple[str, int], ...] = (),
    decimal_precision_histogram: Tuple[float, ...] = (1.0,),
) -> FeatureVector:
    return FeatureVector(
        entity_type_counts=entity_type_counts,
        angle_histogram=angle_histogram,
        length_histogram=length_histogram,
        bounding_box=bounding_box if bounding_box is not None else _bounding_box(),
        spatial_density=spatial_density,
        layer_entity_counts=layer_entity_counts,
        block_usage_counts=block_usage_counts,
        decimal_precision_histogram=decimal_precision_histogram,
    )


def _graph_metrics(
    degree: Tuple[Tuple[int, float], ...] = ((0, 1.0), (1, 1.0)),
    betweenness: Tuple[Tuple[int, float], ...] = ((0, 0.0), (1, 0.0)),
    clustering: Tuple[Tuple[int, float], ...] = ((0, 0.0), (1, 0.0)),
    component_count: int = 1,
    node_count: int = 2,
    edge_count: int = 1,
) -> GraphMetrics:
    return GraphMetrics(
        degree_centrality=degree,
        betweenness_centrality=betweenness,
        clustering_coefficient=clustering,
        component_count=component_count,
        node_count=node_count,
        edge_count=edge_count,
    )


def _sequence_result(combined_score: float) -> SequenceComparisonResult:
    return SequenceComparisonResult(
        sequence_a_length=4,
        sequence_b_length=4,
        lcs_length=4,
        lcs_ratio=combined_score,
        levenshtein_distance=0,
        levenshtein_ratio=combined_score,
        combined_score=combined_score,
        dtw_distance=None,
    )


def _metadata(
    author: str = "Maria",
    dxf_version: str = "AC1032",
    last_saved_by: str = "Maria",
) -> DocumentMetadata:
    return DocumentMetadata(
        author=author,
        dxf_version=dxf_version,
        last_saved_by=last_saved_by,
        created=datetime(2026, 1, 1),
        modified=datetime(2026, 1, 2),
    )


def _document(
    layers: Tuple[str, ...] = ("0", "COTAS"),
    blocks: Tuple[str, ...] = ("BLOCO_A",),
    text_styles: Tuple[str, ...] = ("Standard",),
    dimension_styles: Tuple[str, ...] = ("Standard",),
    metadata: Optional[DocumentMetadata] = None,
) -> CadDocument:
    return CadDocument(
        source_path=Path("sample.dxf"),
        entities=(),
        layers=layers,
        blocks=blocks,
        text_styles=text_styles,
        dimension_styles=dimension_styles,
        metadata=metadata if metadata is not None else _metadata(),
    )


def test_compare_geometry_identical_vectors_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    feature = _feature_vector()
    assert engine.compare_geometry(feature, feature) == pytest.approx(1.0)


def test_compare_geometry_completely_different_vectors_yields_zero(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(
        entity_type_counts=(("LINE", 10),),
        angle_histogram=(1.0, 0.0),
        length_histogram=(1.0, 0.0),
        bounding_box=_bounding_box(10.0, 10.0),
        spatial_density=(1.0, 0.0, 0.0, 0.0),
    )
    feature_b = _feature_vector(
        entity_type_counts=(("CIRCLE", 10),),
        angle_histogram=(0.0, 1.0),
        length_histogram=(0.0, 1.0),
        bounding_box=_bounding_box(0.0, 0.0),
        spatial_density=(0.0, 0.0, 0.0, 1.0),
    )
    assert engine.compare_geometry(feature_a, feature_b) == pytest.approx(0.0)


def test_compare_geometry_mismatched_histogram_lengths_yields_zero(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(angle_histogram=(1.0, 0.0))
    feature_b = _feature_vector(angle_histogram=(0.5, 0.25, 0.25))
    assert engine.compare_geometry(feature_a, feature_b) < 1.0


def test_compare_sequence_returns_combined_score(engine: SimilarityEngine) -> None:
    assert engine.compare_sequence(_sequence_result(0.75)) == pytest.approx(0.75)


def test_compare_graph_identical_metrics_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    metrics = _graph_metrics()
    assert engine.compare_graph(metrics, metrics) == pytest.approx(1.0)


def test_compare_graph_different_density_lowers_score(engine: SimilarityEngine) -> None:
    dense = _graph_metrics(node_count=4, edge_count=6, component_count=1)
    sparse = _graph_metrics(node_count=4, edge_count=1, component_count=3)
    score = engine.compare_graph(dense, sparse)
    assert 0.0 <= score < 1.0


def test_compare_graph_single_node_has_zero_density(engine: SimilarityEngine) -> None:
    single_node = _graph_metrics(node_count=1, edge_count=0, component_count=1)
    assert engine.compare_graph(single_node, single_node) == pytest.approx(1.0)


def test_compare_styles_identical_documents_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    document = _document()
    assert engine.compare_styles(document, document) == pytest.approx(1.0)


def test_compare_styles_disjoint_layers_lowers_score(engine: SimilarityEngine) -> None:
    document_a = _document(layers=("0", "COTAS"))
    document_b = _document(layers=("EIXO", "HACHURA"))
    assert engine.compare_styles(document_a, document_b) < 1.0


def test_compare_metadata_matching_fields_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    document = _document()
    assert engine.compare_metadata(document, document) == pytest.approx(1.0)


def test_compare_metadata_different_authors_lowers_score(engine: SimilarityEngine) -> None:
    document_a = _document(metadata=_metadata(author="Maria"))
    document_b = _document(metadata=_metadata(author="Joao"))
    assert engine.compare_metadata(document_a, document_b) < 1.0


def test_compare_metadata_both_authors_missing_is_neutral(engine: SimilarityEngine) -> None:
    document_a = _document(metadata=_metadata(author=None, last_saved_by=None))
    document_b = _document(metadata=_metadata(author=None, last_saved_by=None))
    assert engine.compare_metadata(document_a, document_b) == pytest.approx(2.0 / 3.0)


def test_compute_score_identical_inputs_yields_maximum_score(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    assert isinstance(result, SimilarityResult)
    assert result.total_score == pytest.approx(100.0)


def test_compute_score_component_weights_match_config(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    config = Config()
    for component in result.component_scores:
        assert component.weight == pytest.approx(config.weights[component.name])


def test_compute_score_lookup_by_name(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    geometry = result.component("geometry")
    assert isinstance(geometry, ComponentScore)
    assert geometry.score == pytest.approx(1.0)
    assert "Geometria" in geometry.justification


def test_compute_score_unknown_component_raises_key_error(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    with pytest.raises(KeyError):
        result.component("unknown")


def test_compute_score_respects_custom_weights() -> None:
    config = Config(
        weights={
            "geometry": 1.0,
            "sequence": 0.0,
            "graph": 0.0,
            "styles": 0.0,
            "metadata": 0.0,
        }
    )
    engine = SimilarityEngine(config=config)
    feature = _feature_vector()
    metrics = _graph_metrics()
    document_a = _document(metadata=_metadata(author="Maria"))
    document_b = _document(metadata=_metadata(author="Outro"), layers=("X",))
    result = engine.compute_score(
        feature, feature, _sequence_result(0.0), metrics, metrics, document_a, document_b
    )
    assert result.total_score == pytest.approx(100.0)


def test_compute_score_is_deterministic(engine: SimilarityEngine) -> None:
    feature_a = _feature_vector()
    feature_b = _feature_vector(entity_type_counts=(("LINE", 3), ("CIRCLE", 1)))
    metrics_a = _graph_metrics()
    metrics_b = _graph_metrics(node_count=3, edge_count=2, component_count=1)
    document_a = _document()
    document_b = _document(layers=("0",))
    sequence_result = _sequence_result(0.6)
    first = engine.compute_score(
        feature_a, feature_b, sequence_result, metrics_a, metrics_b, document_a, document_b
    )
    second = engine.compute_score(
        feature_a, feature_b, sequence_result, metrics_a, metrics_b, document_a, document_b
    )
    assert first == second


def test_similarity_result_is_frozen(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    with pytest.raises(Exception):
        result.total_score = 0.0  # type: ignore[misc]


def test_component_score_is_frozen(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    document = _document()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, document, document
    )
    with pytest.raises(Exception):
        result.component_scores[0].score = 0.0  # type: ignore[misc]
