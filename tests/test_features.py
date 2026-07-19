import math
from pathlib import Path
from typing import Tuple

import pytest

from config import Config
from detector.features import (
    BoundingBox,
    FeatureExtractor,
    FeatureVector,
    _decimal_places,
)
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
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


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


def _sample_entities() -> Tuple[CadEntity, ...]:
    return (
        LineEntity(handle="1", layer="GEOMETRY", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        LineEntity(handle="2", layer="GEOMETRY", start=(0.0, 0.0, 0.0), end=(0.0, 5.0, 0.0)),
        ArcEntity(
            handle="3",
            layer="GEOMETRY",
            center=(3.0, 6.0, 0.0),
            radius=1.5,
            start_angle=10.0,
            end_angle=170.0,
        ),
        CircleEntity(handle="4", layer="GEOMETRY", center=(4.0, 1.0, 0.0), radius=0.8),
        PolylineEntity(
            handle="5",
            layer="GEOMETRY",
            points=((1.0, 1.0), (2.0, 3.0), (5.0, 0.5)),
            closed=True,
        ),
        InsertEntity(
            handle="6",
            layer="BLOCKS",
            block_name="PARAFUSO",
            insert_point=(6.0, 4.0, 0.0),
            x_scale=1.0,
            y_scale=1.0,
            z_scale=1.0,
            rotation=30.0,
        ),
        InsertEntity(
            handle="7",
            layer="BLOCKS",
            block_name="PARAFUSO",
            insert_point=(2.0, 2.0, 0.0),
            x_scale=1.0,
            y_scale=1.0,
            z_scale=1.0,
            rotation=90.0,
        ),
        TextEntity(
            handle="8",
            layer="TEXT",
            text="hi",
            insert_point=(2.0, 8.0, 0.0),
            height=0.5,
            style="Standard",
        ),
        MTextEntity(
            handle="9",
            layer="TEXT",
            text="hi",
            insert_point=(8.0, 5.0, 0.0),
            char_height=0.5,
            style="Standard",
        ),
        DimensionEntity(
            handle="10",
            layer="DIMENSIONS",
            dim_type=0,
            style="Standard",
            text_override="<>",
            measurement=10.0,
        ),
    )


def test_extract_returns_feature_vector(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert isinstance(result, FeatureVector)


def test_feature_vector_is_frozen(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    with pytest.raises(Exception):
        result.entity_type_counts = ()  # type: ignore[misc]


def test_entity_type_counts_include_all_known_types(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    counts = dict(result.entity_type_counts)
    assert counts["LINE"] == 2
    assert counts["ARC"] == 1
    assert counts["CIRCLE"] == 1
    assert counts["LWPOLYLINE"] == 1
    assert counts["INSERT"] == 2
    assert counts["TEXT"] == 1
    assert counts["MTEXT"] == 1
    assert counts["DIMENSION"] == 1


def test_entity_type_counts_are_deterministic_across_documents(
    extractor: FeatureExtractor,
) -> None:
    empty_document = _document(())
    result = extractor.extract(empty_document)
    counts = dict(result.entity_type_counts)
    assert counts["LINE"] == 0
    assert set(counts.keys()) == {
        "LINE",
        "ARC",
        "CIRCLE",
        "LWPOLYLINE",
        "INSERT",
        "TEXT",
        "MTEXT",
        "DIMENSION",
    }


def test_angle_histogram_has_configured_bin_count(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.angle_histogram) == Config().angle_bins


def test_angle_histogram_sums_to_one_when_samples_exist(
    extractor: FeatureExtractor,
) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert sum(result.angle_histogram) == pytest.approx(1.0)


def test_angle_histogram_is_all_zero_without_angle_data(
    extractor: FeatureExtractor,
) -> None:
    document = _document(
        (CircleEntity(handle="1", layer="0", center=(0.0, 0.0, 0.0), radius=1.0),)
    )
    result = extractor.extract(document)
    assert sum(result.angle_histogram) == pytest.approx(0.0)


def test_horizontal_line_falls_into_zero_degree_bin(extractor: FeatureExtractor) -> None:
    document = _document(
        (LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0)),)
    )
    result = extractor.extract(document)
    assert result.angle_histogram[0] == pytest.approx(1.0)


def test_length_histogram_has_configured_bin_count(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.length_histogram) == Config().length_bins


def test_length_histogram_sums_to_one_when_samples_exist(
    extractor: FeatureExtractor,
) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert sum(result.length_histogram) == pytest.approx(1.0)


def test_length_histogram_is_all_zero_without_length_data(
    extractor: FeatureExtractor,
) -> None:
    document = _document(())
    result = extractor.extract(document)
    assert sum(result.length_histogram) == pytest.approx(0.0)
    assert len(result.length_histogram) == Config().length_bins


def test_longest_line_falls_into_last_length_bin(extractor: FeatureExtractor) -> None:
    document = _document(
        (
            LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0)),
            LineEntity(handle="2", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),
        )
    )
    result = extractor.extract(document)
    assert result.length_histogram[-1] > 0.0


def test_bounding_box_matches_extreme_points(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    box = result.bounding_box
    assert isinstance(box, BoundingBox)
    assert box.max_x >= box.min_x
    assert box.max_y >= box.min_y
    assert box.width == pytest.approx(box.max_x - box.min_x)
    assert box.height == pytest.approx(box.max_y - box.min_y)
    assert box.area == pytest.approx(box.width * box.height)


def test_bounding_box_is_all_zero_for_empty_document(
    extractor: FeatureExtractor,
) -> None:
    document = _document(())
    result = extractor.extract(document)
    box = result.bounding_box
    assert box == BoundingBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_bounding_box_includes_circle_extremes() -> None:
    extractor = FeatureExtractor()
    document = _document(
        (CircleEntity(handle="1", layer="0", center=(5.0, 5.0, 0.0), radius=2.0),)
    )
    result = extractor.extract(document)
    box = result.bounding_box
    assert box.min_x == pytest.approx(3.0)
    assert box.max_x == pytest.approx(7.0)
    assert box.min_y == pytest.approx(3.0)
    assert box.max_y == pytest.approx(7.0)


def test_spatial_density_has_grid_squared_bins(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.spatial_density) == Config().spatial_grid ** 2


def test_spatial_density_sums_to_one_when_points_exist(
    extractor: FeatureExtractor,
) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert sum(result.spatial_density) == pytest.approx(1.0)


def test_spatial_density_is_all_zero_for_empty_document(
    extractor: FeatureExtractor,
) -> None:
    document = _document(())
    result = extractor.extract(document)
    assert sum(result.spatial_density) == pytest.approx(0.0)


def test_spatial_density_uses_custom_grid_size() -> None:
    config = Config(spatial_grid=4)
    extractor = FeatureExtractor(config=config)
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.spatial_density) == 16


def test_layer_entity_counts_reflect_actual_usage(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    counts = dict(result.layer_entity_counts)
    assert counts["GEOMETRY"] == 5
    assert counts["BLOCKS"] == 2
    assert counts["TEXT"] == 2
    assert counts["DIMENSIONS"] == 1


def test_layer_entity_counts_empty_for_empty_document(
    extractor: FeatureExtractor,
) -> None:
    document = _document(())
    result = extractor.extract(document)
    assert result.layer_entity_counts == ()


def test_block_usage_counts_reflect_insert_references(
    extractor: FeatureExtractor,
) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    counts = dict(result.block_usage_counts)
    assert counts["PARAFUSO"] == 2


def test_block_usage_counts_empty_without_inserts(extractor: FeatureExtractor) -> None:
    document = _document(
        (LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0)),)
    )
    result = extractor.extract(document)
    assert result.block_usage_counts == ()


def test_decimal_precision_histogram_has_fixed_bin_count(
    extractor: FeatureExtractor,
) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.decimal_precision_histogram) == 7


def test_decimal_precision_histogram_sums_to_one(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert sum(result.decimal_precision_histogram) == pytest.approx(1.0)


def test_decimal_precision_histogram_detects_integer_values(
    extractor: FeatureExtractor,
) -> None:
    document = _document(
        (LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0)),)
    )
    result = extractor.extract(document)
    assert result.decimal_precision_histogram[0] == pytest.approx(1.0)


def test_decimal_precision_histogram_detects_two_decimal_places(
    extractor: FeatureExtractor,
) -> None:
    document = _document(
        (
            LineEntity(
                handle="1", layer="0", start=(0.12, 0.0, 0.0), end=(10.34, 0.0, 0.0)
            ),
        )
    )
    result = extractor.extract(document)
    assert result.decimal_precision_histogram[2] > 0.0


def test_decimal_places_clamps_to_max() -> None:
    assert _decimal_places(math.pi, max_places=6) == 6


def test_decimal_places_detects_zero_places() -> None:
    assert _decimal_places(4.0, max_places=6) == 0


def test_decimal_places_detects_exact_places() -> None:
    assert _decimal_places(1.25, max_places=6) == 2


def test_extract_is_reproducible(extractor: FeatureExtractor) -> None:
    document = _document(_sample_entities())
    first = extractor.extract(document)
    second = extractor.extract(document)
    assert first == second


def test_extractor_uses_injected_config() -> None:
    config = Config(angle_bins=8, length_bins=4, spatial_grid=2)
    extractor = FeatureExtractor(config=config)
    document = _document(_sample_entities())
    result = extractor.extract(document)
    assert len(result.angle_histogram) == 8
    assert len(result.length_histogram) == 4
    assert len(result.spatial_density) == 4
