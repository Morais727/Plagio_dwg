from pathlib import Path
from typing import Tuple

import pytest

from config import Config
from detector.graph import GraphBuilder, GraphMetrics
from detector.reader import (
    ArcEntity,
    CadDocument,
    CadEntity,
    CircleEntity,
    DimensionEntity,
    DocumentMetadata,
    InsertEntity,
    LineEntity,
    MTextEntity,
    PolylineEntity,
    TextEntity,
)


@pytest.fixture
def builder() -> GraphBuilder:
    return GraphBuilder()


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        author=None,
        dxf_version="AC1032",
        last_saved_by=None,
        created=None,
        modified=None,
    )


def _document(entities: Tuple[CadEntity, ...]) -> CadDocument:
    return CadDocument(
        source_path=Path("sample.dxf"),
        entities=entities,
        layers=("0",),
        blocks=(),
        text_styles=("Standard",),
        dimension_styles=("Standard",),
        metadata=_metadata(),
    )


def test_build_creates_one_node_per_entity(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0)),
        CircleEntity(handle="2", layer="0", center=(5.0, 5.0, 0.0), radius=1.0),
    )
    graph = builder.build(_document(entities))
    assert graph.number_of_nodes() == 2
    assert graph.nodes[0]["handle"] == "1"
    assert graph.nodes[0]["entity_type"] == "LineEntity"
    assert graph.nodes[1]["entity_type"] == "CircleEntity"


def test_perpendicular_lines_sharing_endpoint_intersect_and_are_perpendicular(
    builder: GraphBuilder,
) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(0.0, 0.0, 0.0), end=(0.0, 10.0, 0.0)),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    relations = set(graph.edges[0, 1]["relations"])
    assert "intersection" in relations
    assert "perpendicular" in relations


def test_crossing_lines_intersect_and_are_perpendicular(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 10.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(0.0, 10.0, 0.0), end=(10.0, 0.0, 0.0)),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    relations = set(graph.edges[0, 1]["relations"])
    assert "intersection" in relations
    assert "perpendicular" in relations


def test_parallel_non_intersecting_lines_are_flagged_parallel_only(
    builder: GraphBuilder,
) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(0.0, 5.0, 0.0), end=(10.0, 5.0, 0.0)),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    relations = set(graph.edges[0, 1]["relations"])
    assert relations == {"parallel"}


def test_unrelated_lines_produce_no_edge(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.3, 0.0)),
        LineEntity(handle="2", layer="0", start=(50.0, 50.0, 0.0), end=(60.0, 55.0, 0.0)),
    )
    graph = builder.build(_document(entities))
    assert not graph.has_edge(0, 1)


def test_line_tangent_to_circle_is_flagged_tangency(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        CircleEntity(handle="2", layer="0", center=(5.0, 1.0, 0.0), radius=1.0),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    assert "tangency" in graph.edges[0, 1]["relations"]


def test_line_crossing_circle_is_flagged_intersection(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        CircleEntity(handle="2", layer="0", center=(5.0, 0.5, 0.0), radius=1.0),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    assert "intersection" in graph.edges[0, 1]["relations"]


def test_externally_tangent_circles_are_flagged_tangency(builder: GraphBuilder) -> None:
    entities = (
        CircleEntity(handle="1", layer="0", center=(0.0, 0.0, 0.0), radius=1.0),
        CircleEntity(handle="2", layer="0", center=(3.0, 0.0, 0.0), radius=2.0),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    assert "tangency" in graph.edges[0, 1]["relations"]


def test_overlapping_circles_are_flagged_intersection(builder: GraphBuilder) -> None:
    entities = (
        CircleEntity(handle="1", layer="0", center=(0.0, 0.0, 0.0), radius=2.0),
        CircleEntity(handle="2", layer="0", center=(3.0, 0.0, 0.0), radius=2.0),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    assert "intersection" in graph.edges[0, 1]["relations"]


def test_disjoint_circles_produce_no_edge(builder: GraphBuilder) -> None:
    entities = (
        CircleEntity(handle="1", layer="0", center=(0.0, 0.0, 0.0), radius=1.0),
        CircleEntity(handle="2", layer="0", center=(10.0, 10.0, 0.0), radius=1.0),
    )
    graph = builder.build(_document(entities))
    assert not graph.has_edge(0, 1)


def test_polyline_segment_can_intersect_another_entity(builder: GraphBuilder) -> None:
    entities = (
        PolylineEntity(
            handle="1",
            layer="0",
            points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
            closed=False,
        ),
        LineEntity(handle="2", layer="0", start=(5.0, -5.0, 0.0), end=(5.0, 5.0, 0.0)),
    )
    graph = builder.build(_document(entities))
    assert graph.has_edge(0, 1)
    assert "intersection" in graph.edges[0, 1]["relations"]


def test_non_geometric_entities_stay_isolated(builder: GraphBuilder) -> None:
    entities = (
        TextEntity(
            handle="1", layer="0", text="hi", insert_point=(0.0, 0.0, 0.0), height=0.5, style="Standard"
        ),
        MTextEntity(
            handle="2", layer="0", text="hi", insert_point=(1.0, 0.0, 0.0), char_height=0.5, style="Standard"
        ),
        DimensionEntity(
            handle="3", layer="0", dim_type=0, style="Standard", text_override="<>", measurement=1.0
        ),
        InsertEntity(
            handle="4",
            layer="0",
            block_name="X",
            insert_point=(2.0, 0.0, 0.0),
            x_scale=1.0,
            y_scale=1.0,
            z_scale=1.0,
            rotation=0.0,
        ),
        ArcEntity(
            handle="5", layer="0", center=(100.0, 100.0, 0.0), radius=1.0, start_angle=0.0, end_angle=90.0
        ),
    )
    graph = builder.build(_document(entities))
    assert graph.number_of_edges() == 0
    assert graph.number_of_nodes() == 5


def test_builder_uses_injected_config_for_parallel_tolerance() -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(0.0, 5.0, 0.0), end=(10.0, 5.52408, 0.0)),
    )
    document = _document(entities)

    strict_graph = GraphBuilder(config=Config()).build(document)
    assert not strict_graph.has_edge(0, 1)

    loose_config = Config(graph_parallel_tolerance_degrees=5.0)
    loose_graph = GraphBuilder(config=loose_config).build(document)
    assert loose_graph.has_edge(0, 1)
    assert "parallel" in loose_graph.edges[0, 1]["relations"]


def _triangle_document() -> CadDocument:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(10.0, 0.0, 0.0), end=(5.0, 8.0, 0.0)),
        LineEntity(handle="3", layer="0", start=(5.0, 8.0, 0.0), end=(0.0, 0.0, 0.0)),
    )
    return _document(entities)


def test_compute_metrics_returns_entry_for_every_node(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    assert isinstance(metrics, GraphMetrics)
    assert len(metrics.degree_centrality) == 3
    assert len(metrics.betweenness_centrality) == 3
    assert len(metrics.clustering_coefficient) == 3
    assert metrics.node_count == 3
    assert metrics.edge_count == 3


def test_triangle_has_a_single_component(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    assert metrics.component_count == 1


def test_triangle_degree_centrality_is_maximal(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    for _, value in metrics.degree_centrality:
        assert value == pytest.approx(1.0)


def test_triangle_clustering_coefficient_is_one(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    for _, value in metrics.clustering_coefficient:
        assert value == pytest.approx(1.0)


def test_triangle_betweenness_centrality_is_zero(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    for _, value in metrics.betweenness_centrality:
        assert value == pytest.approx(0.0)


def test_disconnected_entities_produce_multiple_components(builder: GraphBuilder) -> None:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(50.0, 50.0, 0.0), end=(51.0, 50.7, 0.0)),
    )
    graph = builder.build(_document(entities))
    metrics = builder.compute_metrics(graph)
    assert metrics.component_count == 2


def test_build_is_deterministic(builder: GraphBuilder) -> None:
    document = _triangle_document()
    first = builder.build(document)
    second = builder.build(document)
    assert sorted(first.edges(data=True)) == sorted(second.edges(data=True))
    assert sorted(first.nodes(data=True)) == sorted(second.nodes(data=True))


def test_graph_metrics_is_frozen(builder: GraphBuilder) -> None:
    graph = builder.build(_triangle_document())
    metrics = builder.compute_metrics(graph)
    with pytest.raises(Exception):
        metrics.node_count = 99  # type: ignore[misc]
