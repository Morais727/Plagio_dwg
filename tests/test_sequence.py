from pathlib import Path
from typing import Tuple

import pytest

from config import Config
from detector.sequence import SequenceAnalyzer, SequenceComparisonResult, entity_code
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
def analyzer() -> SequenceAnalyzer:
    return SequenceAnalyzer()


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


def _line() -> LineEntity:
    return LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))


def _arc() -> ArcEntity:
    return ArcEntity(
        handle="2", layer="0", center=(0.0, 0.0, 0.0), radius=1.0, start_angle=0.0, end_angle=90.0
    )


def _circle() -> CircleEntity:
    return CircleEntity(handle="3", layer="0", center=(0.0, 0.0, 0.0), radius=1.0)


def _polyline() -> PolylineEntity:
    return PolylineEntity(handle="4", layer="0", points=((0.0, 0.0), (1.0, 0.0)), closed=False)


def _insert() -> InsertEntity:
    return InsertEntity(
        handle="5",
        layer="0",
        block_name="X",
        insert_point=(0.0, 0.0, 0.0),
        x_scale=1.0,
        y_scale=1.0,
        z_scale=1.0,
        rotation=0.0,
    )


def _text() -> TextEntity:
    return TextEntity(
        handle="6", layer="0", text="hi", insert_point=(0.0, 0.0, 0.0), height=0.5, style="Standard"
    )


def _mtext() -> MTextEntity:
    return MTextEntity(
        handle="7", layer="0", text="hi", insert_point=(0.0, 0.0, 0.0), char_height=0.5, style="Standard"
    )


def _dimension() -> DimensionEntity:
    return DimensionEntity(
        handle="8", layer="0", dim_type=0, style="Standard", text_override="<>", measurement=1.0
    )


def test_entity_code_maps_every_known_entity_type() -> None:
    assert entity_code(_line()) == "L"
    assert entity_code(_arc()) == "A"
    assert entity_code(_circle()) == "C"
    assert entity_code(_polyline()) == "P"
    assert entity_code(_insert()) == "I"
    assert entity_code(_text()) == "T"
    assert entity_code(_mtext()) == "M"
    assert entity_code(_dimension()) == "D"


def test_build_sequence_preserves_entity_order(analyzer: SequenceAnalyzer) -> None:
    entities = (_line(), _line(), _line(), _circle(), _circle(), _arc(), _dimension(), _text())
    document = _document(entities)
    assert analyzer.build_sequence(document) == "LLLCCADT"


def test_build_sequence_empty_document(analyzer: SequenceAnalyzer) -> None:
    assert analyzer.build_sequence(_document(())) == ""


def test_compare_identical_sequences_yields_perfect_scores(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LLLCCADT", "LLLCCADT")
    assert isinstance(result, SequenceComparisonResult)
    assert result.lcs_length == 8
    assert result.lcs_ratio == pytest.approx(1.0)
    assert result.levenshtein_distance == 0
    assert result.levenshtein_ratio == pytest.approx(1.0)
    assert result.combined_score == pytest.approx(1.0)


def test_compare_completely_different_sequences_yields_low_scores(
    analyzer: SequenceAnalyzer,
) -> None:
    result = analyzer.compare("LLLL", "CCCC")
    assert result.lcs_length == 0
    assert result.lcs_ratio == pytest.approx(0.0)
    assert result.levenshtein_distance == 4
    assert result.levenshtein_ratio == pytest.approx(0.0)
    assert result.combined_score == pytest.approx(0.0)


def test_compare_empty_sequences_yields_perfect_scores(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("", "")
    assert result.sequence_a_length == 0
    assert result.sequence_b_length == 0
    assert result.lcs_length == 0
    assert result.lcs_ratio == pytest.approx(1.0)
    assert result.levenshtein_distance == 0
    assert result.levenshtein_ratio == pytest.approx(1.0)


def test_compare_partial_overlap_known_values(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LCAAT", "LCAT")
    assert result.sequence_a_length == 5
    assert result.sequence_b_length == 4
    assert result.lcs_length == 4
    assert result.levenshtein_distance == 1


def test_compare_reports_input_lengths(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LLCA", "LC")
    assert result.sequence_a_length == 4
    assert result.sequence_b_length == 2


def test_compare_combined_score_respects_config_weights() -> None:
    lcs_only = SequenceAnalyzer(config=Config(lcs_weight=1.0, levenshtein_weight=0.0))
    result = lcs_only.compare("LCAT", "LCXT")
    assert result.combined_score == pytest.approx(result.lcs_ratio)

    levenshtein_only = SequenceAnalyzer(config=Config(lcs_weight=0.0, levenshtein_weight=1.0))
    result = levenshtein_only.compare("LCAT", "LCXT")
    assert result.combined_score == pytest.approx(result.levenshtein_ratio)


def test_compare_without_dtw_leaves_field_none(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LCAT", "LCAT")
    assert result.dtw_distance is None


def test_compare_with_dtw_populates_field(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LCAT", "LCAT", include_dtw=True)
    assert result.dtw_distance == pytest.approx(0.0)


def test_dtw_identical_sequences_is_zero(analyzer: SequenceAnalyzer) -> None:
    assert analyzer.dynamic_time_warping("LLCAT", "LLCAT") == pytest.approx(0.0)


def test_dtw_is_symmetric(analyzer: SequenceAnalyzer) -> None:
    a, b = "LLCAAT", "LCADT"
    assert analyzer.dynamic_time_warping(a, b) == pytest.approx(
        analyzer.dynamic_time_warping(b, a)
    )


def test_dtw_against_empty_sequence_equals_other_length(analyzer: SequenceAnalyzer) -> None:
    assert analyzer.dynamic_time_warping("", "") == pytest.approx(0.0)
    assert analyzer.dynamic_time_warping("LCAT", "") == pytest.approx(4.0)
    assert analyzer.dynamic_time_warping("", "LCAT") == pytest.approx(4.0)


def test_dtw_more_similar_sequences_have_lower_distance(analyzer: SequenceAnalyzer) -> None:
    close = analyzer.dynamic_time_warping("LLLCCADT", "LLLCCADT")
    far = analyzer.dynamic_time_warping("LLLCCADT", "TTTTTTTT")
    assert close < far


def test_sequence_comparison_result_is_frozen(analyzer: SequenceAnalyzer) -> None:
    result = analyzer.compare("LCAT", "LCAT")
    with pytest.raises(Exception):
        result.combined_score = 0.0  # type: ignore[misc]


def test_compare_is_deterministic(analyzer: SequenceAnalyzer) -> None:
    first = analyzer.compare("LLLCCADT", "LLCADTT", include_dtw=True)
    second = analyzer.compare("LLLCCADT", "LLCADTT", include_dtw=True)
    assert first == second


def test_build_sequence_is_deterministic(analyzer: SequenceAnalyzer) -> None:
    document = _document((_line(), _arc(), _circle(), _dimension()))
    assert analyzer.build_sequence(document) == analyzer.build_sequence(document)
