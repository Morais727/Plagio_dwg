import math
from datetime import datetime
from typing import Sequence, Tuple

import pytest

from detector.pieces import compare_piece_sets, extract_pieces
from detector.reader import CadDocument, DimensionEntity, DocumentMetadata, LineEntity, MTextEntity


Point2D = Tuple[float, float]

_TOLERANCE = 1e-3


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        author="Aluno A",
        dxf_version="AC1027",
        last_saved_by="Aluno A",
        created=datetime(2026, 1, 1, 12, 0, 0),
        modified=datetime(2026, 1, 1, 12, 0, 0),
    )


def _document(entities: Sequence) -> CadDocument:
    return CadDocument(
        source_path=None,
        entities=tuple(entities),
        blocks=(),
        text_styles=("Standard",),
        dimension_styles=("Standard",),
        metadata=_metadata(),
    )


def _square_lines(
    origin: Point2D = (0.0, 0.0),
    size: float = 10.0,
    angle_degrees: float = 0.0,
) -> Tuple[LineEntity, ...]:
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    transformed = [
        (
            x * cos_a - y * sin_a + origin[0],
            x * sin_a + y * cos_a + origin[1],
        )
        for x, y in corners
    ]
    lines = []
    for index, (start, end) in enumerate(
        zip(transformed, transformed[1:] + transformed[:1])
    ):
        lines.append(
            LineEntity(
                handle=str(index),
                start=(start[0], start[1], 0.0),
                end=(end[0], end[1], 0.0),
                linewidth=0,
            )
        )
    return tuple(lines)


def _triangle_lines(origin: Point2D = (100.0, 100.0), size: float = 6.0) -> Tuple[LineEntity, ...]:
    corners = [(0.0, 0.0), (size, 0.0), (size / 2.0, size)]
    transformed = [(x + origin[0], y + origin[1]) for x, y in corners]
    lines = []
    for index, (start, end) in enumerate(
        zip(transformed, transformed[1:] + transformed[:1])
    ):
        lines.append(
            LineEntity(
                handle=f"t{index}",
                start=(start[0], start[1], 0.0),
                end=(end[0], end[1], 0.0),
                linewidth=0,
            )
        )
    return tuple(lines)


def test_extract_pieces_groups_connected_entities_into_one_piece() -> None:
    document = _document(_square_lines())
    pieces = extract_pieces(document, _TOLERANCE)
    assert len(pieces) == 1
    assert pieces[0].entity_count == 4


def test_extract_pieces_separates_disjoint_shapes() -> None:
    entities = _square_lines() + _triangle_lines()
    document = _document(entities)
    pieces = extract_pieces(document, _TOLERANCE)
    assert len(pieces) == 2


def test_compare_piece_sets_identical_shape_yields_high_score() -> None:
    pieces_a = extract_pieces(_document(_square_lines()), _TOLERANCE)
    pieces_b = extract_pieces(_document(_square_lines()), _TOLERANCE)
    score = compare_piece_sets(pieces_a, pieces_b)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_compare_piece_sets_rotated_translated_shape_stays_high() -> None:
    pieces_a = extract_pieces(
        _document(_square_lines(origin=(0.0, 0.0), size=10.0, angle_degrees=0.0)), _TOLERANCE
    )
    pieces_b = extract_pieces(
        _document(_square_lines(origin=(53.0, -21.0), size=10.0, angle_degrees=37.0)),
        _TOLERANCE,
    )
    score = compare_piece_sets(pieces_a, pieces_b)
    assert score is not None
    assert score > 0.85


def test_compare_piece_sets_different_shape_scores_lower_than_identical() -> None:
    pieces_a = extract_pieces(_document(_square_lines()), _TOLERANCE)
    pieces_b_different = extract_pieces(
        _document(_triangle_lines(origin=(0.0, 0.0), size=10.0)), _TOLERANCE
    )
    pieces_b_identical = extract_pieces(_document(_square_lines()), _TOLERANCE)

    different_score = compare_piece_sets(pieces_a, pieces_b_different)
    identical_score = compare_piece_sets(pieces_a, pieces_b_identical)

    assert different_score is not None
    assert different_score < identical_score


def test_compare_piece_sets_different_piece_counts_applies_coverage_penalty() -> None:
    pieces_a = extract_pieces(
        _document(_square_lines() + _triangle_lines()), _TOLERANCE
    )
    pieces_b = extract_pieces(_document(_square_lines()), _TOLERANCE)
    score = compare_piece_sets(pieces_a, pieces_b)
    assert score is not None
    assert score < 1.0


def test_compare_piece_sets_both_empty_is_not_applicable() -> None:
    assert compare_piece_sets((), ()) is None


def test_compare_piece_sets_one_empty_yields_zero() -> None:
    pieces = extract_pieces(_document(_square_lines()), _TOLERANCE)
    assert compare_piece_sets(pieces, ()) == pytest.approx(0.0)


def _annotation_only_entities() -> Tuple:
    return (
        MTextEntity(
            handle="m1", text="nota", insert_point=(50.0, 50.0, 0.0),
            char_height=0.2, style="Standard", linewidth=0,
        ),
        DimensionEntity(
            handle="d1", insert_point=(60.0, 60.0, 0.0), dim_type=0, style="Standard",
            text_override="", measurement=5.0, linewidth=0,
        ),
    )


def test_extract_pieces_treats_pointless_entities_as_pieces_with_no_shape() -> None:
    document = _document(_annotation_only_entities())
    pieces = extract_pieces(document, _TOLERANCE)
    assert len(pieces) == 2
    assert all(len(piece.pca_points) == 0 for piece in pieces)


def test_compare_piece_sets_identical_document_with_annotation_only_entities_is_perfect() -> None:
    entities = _square_lines() + _annotation_only_entities()
    pieces_a = extract_pieces(_document(entities), _TOLERANCE)
    pieces_b = extract_pieces(_document(entities), _TOLERANCE)
    assert compare_piece_sets(pieces_a, pieces_b) == pytest.approx(1.0, abs=1e-9)


def test_compare_piece_sets_annotation_only_pieces_differ_by_type() -> None:
    pieces_a = extract_pieces(_document((_annotation_only_entities()[0],)), _TOLERANCE)
    pieces_b = extract_pieces(_document((_annotation_only_entities()[1],)), _TOLERANCE)
    score = compare_piece_sets(pieces_a, pieces_b)
    assert score is not None
    assert score < 1.0
