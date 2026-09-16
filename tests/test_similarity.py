from typing import Optional, Tuple

import pytest

from config import Config
from detector.features import BoundingBox, FeatureVector
from detector.graph import GraphMetrics
from detector.reader import CadDocument, DocumentMetadata, LineEntity
from detector.sequence import SequenceComparisonResult, TextSequenceComparisonResult
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
    block_usage_counts: Tuple[Tuple[str, int], ...] = (),
    decimal_precision_histogram: Tuple[float, ...] = (1.0,),
    dimension_distance_histogram: Tuple[float, ...] = (1.0,),
    text_positions: Tuple[Tuple[str, float, float], ...] = (),
) -> FeatureVector:
    return FeatureVector(
        entity_type_counts=entity_type_counts,
        angle_histogram=angle_histogram,
        length_histogram=length_histogram,
        bounding_box=bounding_box if bounding_box is not None else _bounding_box(),
        spatial_density=spatial_density,
        block_usage_counts=block_usage_counts,
        decimal_precision_histogram=decimal_precision_histogram,
        dimension_distance_histogram=dimension_distance_histogram,
        text_positions=text_positions,
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


def _text_sequence_result(combined_score: float = 1.0) -> TextSequenceComparisonResult:
    return TextSequenceComparisonResult(
        text_a_length=2,
        text_b_length=2,
        levenshtein_ratio=combined_score,
        combined_score=combined_score,
    )


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        author="Aluno A",
        dxf_version="AC1027",
        last_saved_by="Aluno A",
        created=None,
        modified=None,
    )


def _cad_document(
    text_styles: Tuple[str, ...] = ("Standard",),
    dimension_styles: Tuple[str, ...] = ("Standard",),
) -> CadDocument:
    return CadDocument(
        source_path=None,
        entities=(),
        blocks=(),
        text_styles=text_styles,
        dimension_styles=dimension_styles,
        metadata=_metadata(),
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


def test_compute_score_identical_inputs_yields_maximum_score(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    assert isinstance(result, SimilarityResult)
    assert result.total_score == pytest.approx(100.0)


def test_compute_score_component_weights_match_config(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    config = Config()
    for component in result.component_scores:
        assert component.weight == pytest.approx(config.weights[component.name])


def test_compute_score_lookup_by_name(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    geometry = result.component("geometry")
    assert isinstance(geometry, ComponentScore)
    assert geometry.score == pytest.approx(1.0)
    assert "Geometria" in geometry.justification


def test_compute_score_unknown_component_raises_key_error(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    with pytest.raises(KeyError):
        result.component("unknown")


def test_compute_score_respects_custom_weights() -> None:
    config = Config(
        weights={
            "geometry": 1.0,
            "sequence": 0.0,
            "graph": 0.0,
        }
    )
    engine = SimilarityEngine(config=config)
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(0.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    assert result.total_score == pytest.approx(100.0)


def test_compute_score_is_deterministic(engine: SimilarityEngine) -> None:
    feature_a = _feature_vector()
    feature_b = _feature_vector(entity_type_counts=(("LINE", 3), ("CIRCLE", 1)))
    metrics_a = _graph_metrics()
    metrics_b = _graph_metrics(node_count=3, edge_count=2, component_count=1)
    sequence_result = _sequence_result(0.6)
    first = engine.compute_score(
        feature_a, feature_b, sequence_result, metrics_a, metrics_b, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    second = engine.compute_score(
        feature_a, feature_b, sequence_result, metrics_a, metrics_b, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    assert first == second


def test_similarity_result_is_frozen(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    with pytest.raises(Exception):
        result.total_score = 0.0  # type: ignore[misc]


def test_component_score_is_frozen(engine: SimilarityEngine) -> None:
    feature = _feature_vector()
    metrics = _graph_metrics()
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        _cad_document(), _cad_document(),
    )
    with pytest.raises(Exception):
        result.component_scores[0].score = 0.0  # type: ignore[misc]


def test_compare_blocks_identical_blocks_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    feature = _feature_vector(block_usage_counts=(("PARAFUSO", 3), ("PORCA", 1)))
    assert engine.compare_blocks(feature, feature) == pytest.approx(1.0)


def test_compare_blocks_completely_different_blocks_yields_zero(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(block_usage_counts=(("PARAFUSO", 3),))
    feature_b = _feature_vector(block_usage_counts=(("ARRUELA", 3),))
    assert engine.compare_blocks(feature_a, feature_b) == pytest.approx(0.0)


def test_compare_blocks_partial_overlap_yields_intermediate_score(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(block_usage_counts=(("PARAFUSO", 2), ("PORCA", 1)))
    feature_b = _feature_vector(block_usage_counts=(("PARAFUSO", 2), ("ARRUELA", 1)))
    score = engine.compare_blocks(feature_a, feature_b)
    assert 0.0 < score < 1.0


def test_compare_blocks_no_blocks_in_either_is_not_applicable(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(block_usage_counts=())
    feature_b = _feature_vector(block_usage_counts=())
    assert engine.compare_blocks(feature_a, feature_b) is None


def test_compare_styles_same_styles_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    doc = _cad_document()
    assert engine.compare_styles(doc, doc) == pytest.approx(1.0)


def test_compare_styles_different_styles_yields_zero(
    engine: SimilarityEngine,
) -> None:
    doc_a = _cad_document(text_styles=("Standard",), dimension_styles=("ISO-25",))
    doc_b = _cad_document(text_styles=("Arial",), dimension_styles=("DIN",))
    assert engine.compare_styles(doc_a, doc_b) == pytest.approx(0.0)


def test_compare_decimal_precision_identical_histograms_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    feature = _feature_vector(decimal_precision_histogram=(0.2, 0.3, 0.5))
    assert engine.compare_decimal_precision(feature, feature) == pytest.approx(1.0)


def test_compare_decimal_precision_disjoint_histograms_yields_zero(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(decimal_precision_histogram=(1.0, 0.0))
    feature_b = _feature_vector(decimal_precision_histogram=(0.0, 1.0))
    assert engine.compare_decimal_precision(feature_a, feature_b) == pytest.approx(0.0)


def test_compare_decimal_precision_both_zero_is_not_applicable(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(decimal_precision_histogram=(0.0, 0.0))
    feature_b = _feature_vector(decimal_precision_histogram=(0.0, 0.0))
    assert engine.compare_decimal_precision(feature_a, feature_b) is None


def test_compare_dimensions_both_absent_is_not_applicable(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(dimension_distance_histogram=(0.0, 0.0))
    feature_b = _feature_vector(dimension_distance_histogram=(0.0, 0.0))
    assert engine.compare_dimensions(feature_a, feature_b) is None


def test_compare_dimensions_present_in_one_only_is_dissimilar(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(dimension_distance_histogram=(1.0, 0.0))
    feature_b = _feature_vector(dimension_distance_histogram=(0.0, 0.0))
    assert engine.compare_dimensions(feature_a, feature_b) == pytest.approx(0.0)


def test_compare_styles_both_absent_is_not_applicable(
    engine: SimilarityEngine,
) -> None:
    doc_a = _cad_document(text_styles=(), dimension_styles=())
    doc_b = _cad_document(text_styles=(), dimension_styles=())
    assert engine.compare_styles(doc_a, doc_b) is None


def test_compare_text_both_absent_is_not_applicable(engine: SimilarityEngine) -> None:
    feature_a = _feature_vector(text_positions=())
    feature_b = _feature_vector(text_positions=())
    text_sequence_result = _text_sequence_result(1.0)
    assert engine.compare_text(text_sequence_result, feature_a, feature_b) is None


def test_compare_text_present_in_one_only_is_dissimilar(
    engine: SimilarityEngine,
) -> None:
    feature_a = _feature_vector(text_positions=(("peca 01", 0.0, 0.0),))
    feature_b = _feature_vector(text_positions=())
    text_sequence_result = _text_sequence_result(0.0)
    score = engine.compare_text(text_sequence_result, feature_a, feature_b)
    assert score is not None
    assert score < 0.5


def test_compute_score_renormalizes_when_components_not_applicable(
    engine: SimilarityEngine,
) -> None:
    feature = _feature_vector(
        block_usage_counts=(),
        text_positions=(),
        dimension_distance_histogram=(0.0, 0.0),
        decimal_precision_histogram=(0.0, 0.0),
    )
    metrics = _graph_metrics()
    doc = _cad_document(text_styles=(), dimension_styles=())
    result = engine.compute_score(
        feature, feature, _sequence_result(1.0), metrics, metrics, _text_sequence_result(),
        doc, doc,
    )
    inapplicable = {"blocks", "text", "dimensions", "decimal_precision", "styles", "pieces"}
    for component in result.component_scores:
        assert component.applicable == (component.name not in inapplicable)
    assert result.total_score == pytest.approx(100.0)


def _square_document(origin: Tuple[float, float] = (0.0, 0.0)) -> CadDocument:
    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    points = [(x + origin[0], y + origin[1]) for x, y in corners]
    entities = tuple(
        LineEntity(
            handle=str(index),
            start=(start[0], start[1], 0.0),
            end=(end[0], end[1], 0.0),
            linewidth=0,
        )
        for index, (start, end) in enumerate(zip(points, points[1:] + points[:1]))
    )
    return _cad_document_with_entities(entities)


def _cad_document_with_entities(entities) -> CadDocument:
    return CadDocument(
        source_path=None,
        entities=entities,
        blocks=(),
        text_styles=("Standard",),
        dimension_styles=("Standard",),
        metadata=_metadata(),
    )


def test_compare_pieces_identical_document_yields_perfect_score(
    engine: SimilarityEngine,
) -> None:
    doc = _square_document()
    assert engine.compare_pieces(doc, doc) == pytest.approx(1.0, abs=1e-6)


def test_compare_pieces_no_entities_in_either_is_not_applicable(
    engine: SimilarityEngine,
) -> None:
    empty = _cad_document_with_entities(())
    assert engine.compare_pieces(empty, empty) is None


def test_compare_pieces_present_in_one_only_is_dissimilar(
    engine: SimilarityEngine,
) -> None:
    doc = _square_document()
    empty = _cad_document_with_entities(())
    assert engine.compare_pieces(doc, empty) == pytest.approx(0.0)
