import math
from pathlib import Path
from typing import Tuple

import pytest

from detector.normalizer import Normalizer
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
    Point3D,
    PolylineEntity,
    TextEntity,
)


@pytest.fixture
def normalizer() -> Normalizer:
    return Normalizer()


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
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 2.0, 0.0)),
        LineEntity(handle="2", layer="0", start=(10.0, 2.0, 0.0), end=(7.0, 9.0, 0.0)),
        ArcEntity(
            handle="3",
            layer="0",
            center=(3.0, 6.0, 0.0),
            radius=1.5,
            start_angle=10.0,
            end_angle=170.0,
        ),
        CircleEntity(handle="4", layer="0", center=(4.0, 1.0, 0.0), radius=0.8),
        PolylineEntity(
            handle="5",
            layer="0",
            points=((1.0, 1.0), (2.0, 3.0), (5.0, 0.5)),
            closed=True,
        ),
        InsertEntity(
            handle="6",
            layer="0",
            block_name="B",
            insert_point=(6.0, 4.0, 0.0),
            x_scale=1.0,
            y_scale=1.0,
            z_scale=1.0,
            rotation=30.0,
        ),
        TextEntity(
            handle="7",
            layer="0",
            text="hi",
            insert_point=(2.0, 8.0, 0.0),
            height=0.5,
            style="Standard",
        ),
        MTextEntity(
            handle="8",
            layer="0",
            text="hi",
            insert_point=(8.0, 5.0, 0.0),
            char_height=0.5,
            style="Standard",
        ),
        DimensionEntity(
            handle="9",
            layer="0",
            dim_type=0,
            style="Standard",
            text_override="<>",
            measurement=10.0,
        ),
    )


def _rotate_scale_translate(
    point: Point3D, angle_degrees: float, scale: float, translation: Point3D
) -> Point3D:
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y, z = point
    rx = cos_t * x - sin_t * y
    ry = sin_t * x + cos_t * y
    return (
        rx * scale + translation[0],
        ry * scale + translation[1],
        z * scale + translation[2],
    )


def _transform_entities(
    entities: Tuple[CadEntity, ...],
    angle_degrees: float,
    scale: float,
    translation: Point3D,
) -> Tuple[CadEntity, ...]:
    transformed = []
    for entity in entities:
        if isinstance(entity, LineEntity):
            transformed.append(
                LineEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    start=_rotate_scale_translate(
                        entity.start, angle_degrees, scale, translation
                    ),
                    end=_rotate_scale_translate(
                        entity.end, angle_degrees, scale, translation
                    ),
                )
            )
        elif isinstance(entity, ArcEntity):
            transformed.append(
                ArcEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    center=_rotate_scale_translate(
                        entity.center, angle_degrees, scale, translation
                    ),
                    radius=entity.radius * scale,
                    start_angle=(entity.start_angle + angle_degrees) % 360.0,
                    end_angle=(entity.end_angle + angle_degrees) % 360.0,
                )
            )
        elif isinstance(entity, CircleEntity):
            transformed.append(
                CircleEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    center=_rotate_scale_translate(
                        entity.center, angle_degrees, scale, translation
                    ),
                    radius=entity.radius * scale,
                )
            )
        elif isinstance(entity, PolylineEntity):
            transformed.append(
                PolylineEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    points=tuple(
                        _rotate_scale_translate(
                            (x, y, 0.0), angle_degrees, scale, translation
                        )[:2]
                        for x, y in entity.points
                    ),
                    closed=entity.closed,
                )
            )
        elif isinstance(entity, InsertEntity):
            transformed.append(
                InsertEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    block_name=entity.block_name,
                    insert_point=_rotate_scale_translate(
                        entity.insert_point, angle_degrees, scale, translation
                    ),
                    x_scale=entity.x_scale * scale,
                    y_scale=entity.y_scale * scale,
                    z_scale=entity.z_scale * scale,
                    rotation=(entity.rotation + angle_degrees) % 360.0,
                )
            )
        elif isinstance(entity, TextEntity):
            transformed.append(
                TextEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    text=entity.text,
                    insert_point=_rotate_scale_translate(
                        entity.insert_point, angle_degrees, scale, translation
                    ),
                    height=entity.height * scale,
                    style=entity.style,
                )
            )
        elif isinstance(entity, MTextEntity):
            transformed.append(
                MTextEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    text=entity.text,
                    insert_point=_rotate_scale_translate(
                        entity.insert_point, angle_degrees, scale, translation
                    ),
                    char_height=entity.char_height * scale,
                    style=entity.style,
                )
            )
        elif isinstance(entity, DimensionEntity):
            transformed.append(
                DimensionEntity(
                    handle=entity.handle,
                    layer=entity.layer,
                    dim_type=entity.dim_type,
                    style=entity.style,
                    text_override=entity.text_override,
                    measurement=(
                        entity.measurement * scale
                        if entity.measurement is not None
                        else None
                    ),
                )
            )
    return tuple(transformed)


def _angle_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def test_normalize_centers_geometry_bounding_box_at_origin(
    normalizer: Normalizer,
) -> None:
    document = _document(_sample_entities())
    normalized = normalizer.normalize(document)
    xs = []
    ys = []
    for entity in normalized.entities:
        if isinstance(entity, LineEntity):
            xs.extend([entity.start[0], entity.end[0]])
            ys.extend([entity.start[1], entity.end[1]])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    assert (min_x + max_x) / 2.0 == pytest.approx(0.0, abs=1e-6)
    assert (min_y + max_y) / 2.0 == pytest.approx(0.0, abs=1e-6)


def test_normalize_scales_bounding_box_largest_dimension_to_one(
    normalizer: Normalizer,
) -> None:
    document = _document(_sample_entities())
    transform = normalizer.compute_transform(document)
    normalized = normalizer.normalize(document)
    points = []
    for entity in normalized.entities:
        if isinstance(entity, LineEntity):
            points.append(entity.start)
            points.append(entity.end)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    assert max(width, height) == pytest.approx(1.0, abs=1e-6)
    assert transform.scale > 0.0


def test_normalize_empty_document_returns_identity_transform(
    normalizer: Normalizer,
) -> None:
    document = _document(())
    transform = normalizer.compute_transform(document)
    assert transform.translation == (0.0, 0.0, 0.0)
    assert transform.scale == 1.0
    assert transform.rotation_angle_degrees == 0.0


def test_normalize_single_point_document_returns_identity_rotation_and_scale(
    normalizer: Normalizer,
) -> None:
    document = _document(
        (CircleEntity(handle="1", layer="0", center=(5.0, 5.0, 0.0), radius=0.0),)
    )
    transform = normalizer.compute_transform(document)
    assert transform.rotation_matrix == ((1.0, 0.0), (0.0, 1.0))
    assert transform.scale == 1.0


def test_circle_radius_and_center_are_scaled_and_translated(
    normalizer: Normalizer,
) -> None:
    document = _document(_sample_entities())
    normalized = normalizer.normalize(document)
    original_circle = next(
        e for e in document.entities if isinstance(e, CircleEntity)
    )
    normalized_circle = next(
        e for e in normalized.entities if isinstance(e, CircleEntity)
    )
    transform = normalizer.compute_transform(document)
    assert normalized_circle.radius == pytest.approx(
        original_circle.radius * transform.scale
    )


def test_dimension_measurement_is_scaled_and_none_stays_none(
    normalizer: Normalizer,
) -> None:
    entities = _sample_entities() + (
        DimensionEntity(
            handle="10",
            layer="0",
            dim_type=0,
            style="Standard",
            text_override="<>",
            measurement=None,
        ),
    )
    document = _document(entities)
    normalized = normalizer.normalize(document)
    dimensions = [e for e in normalized.entities if isinstance(e, DimensionEntity)]
    measured = next(d for d in dimensions if d.measurement is not None or d.handle == "9")
    assert measured.measurement is not None
    none_measurement = next(d for d in dimensions if d.handle == "10")
    assert none_measurement.measurement is None


def test_normalize_is_invariant_to_translation_scale_and_rotation(
    normalizer: Normalizer,
) -> None:
    base_entities = _sample_entities()
    base_document = _document(base_entities)
    transformed_entities = _transform_entities(
        base_entities,
        angle_degrees=37.0,
        scale=4.2,
        translation=(120.0, -45.0, 0.0),
    )
    transformed_document = _document(transformed_entities)

    normalized_base = normalizer.normalize(base_document)
    normalized_transformed = normalizer.normalize(transformed_document)

    base_lines = [e for e in normalized_base.entities if isinstance(e, LineEntity)]
    transformed_lines = [
        e for e in normalized_transformed.entities if isinstance(e, LineEntity)
    ]
    for base_line, transformed_line in zip(base_lines, transformed_lines):
        for a, b in zip(base_line.start, transformed_line.start):
            assert a == pytest.approx(b, abs=1e-6)
        for a, b in zip(base_line.end, transformed_line.end):
            assert a == pytest.approx(b, abs=1e-6)

    base_circle = next(
        e for e in normalized_base.entities if isinstance(e, CircleEntity)
    )
    transformed_circle = next(
        e for e in normalized_transformed.entities if isinstance(e, CircleEntity)
    )
    assert base_circle.radius == pytest.approx(transformed_circle.radius, abs=1e-6)
    for a, b in zip(base_circle.center, transformed_circle.center):
        assert a == pytest.approx(b, abs=1e-6)

    base_arc = next(e for e in normalized_base.entities if isinstance(e, ArcEntity))
    transformed_arc = next(
        e for e in normalized_transformed.entities if isinstance(e, ArcEntity)
    )
    assert _angle_distance(base_arc.start_angle, transformed_arc.start_angle) < 1e-4
    assert _angle_distance(base_arc.end_angle, transformed_arc.end_angle) < 1e-4

    base_insert = next(
        e for e in normalized_base.entities if isinstance(e, InsertEntity)
    )
    transformed_insert = next(
        e for e in normalized_transformed.entities if isinstance(e, InsertEntity)
    )
    assert _angle_distance(base_insert.rotation, transformed_insert.rotation) < 1e-4
    assert base_insert.x_scale == pytest.approx(
        transformed_insert.x_scale, abs=1e-6
    )


def test_polyline_points_are_transformed(normalizer: Normalizer) -> None:
    document = _document(_sample_entities())
    normalized = normalizer.normalize(document)
    normalized_polyline = next(
        e for e in normalized.entities if isinstance(e, PolylineEntity)
    )
    assert len(normalized_polyline.points) == 3
    assert all(len(point) == 2 for point in normalized_polyline.points)


def test_insert_and_text_entities_preserve_non_geometric_fields(
    normalizer: Normalizer,
) -> None:
    document = _document(_sample_entities())
    normalized = normalizer.normalize(document)
    insert = next(e for e in normalized.entities if isinstance(e, InsertEntity))
    text = next(e for e in normalized.entities if isinstance(e, TextEntity))
    assert insert.block_name == "B"
    assert insert.layer == "0"
    assert text.text == "hi"
    assert text.style == "Standard"


def test_normalize_preserves_document_level_fields(normalizer: Normalizer) -> None:
    document = _document(_sample_entities())
    normalized = normalizer.normalize(document)
    assert normalized.source_path == document.source_path
    assert normalized.layers == document.layers
    assert normalized.blocks == document.blocks
    assert normalized.metadata == document.metadata
    assert len(normalized.entities) == len(document.entities)


def test_normalize_is_reproducible(normalizer: Normalizer) -> None:
    document = _document(_sample_entities())
    first = normalizer.normalize(document)
    second = normalizer.normalize(document)
    assert first == second
