import pytest

from detector.corpus import CorpusContext, build_corpus_context, percentile_rank, z_score


def test_build_corpus_context_computes_mean_and_std() -> None:
    context = build_corpus_context([80.0, 80.0, 80.0, 90.0])
    assert context.mean == pytest.approx(82.5)
    assert context.std == pytest.approx(4.330127, rel=1e-4)


def test_build_corpus_context_empty_scores() -> None:
    context = build_corpus_context([])
    assert context.mean == 0.0
    assert context.std == 0.0
    assert context.scores == ()


def test_z_score_zero_for_mean_value() -> None:
    context = build_corpus_context([70.0, 80.0, 90.0])
    assert z_score(80.0, context) == pytest.approx(0.0)


def test_z_score_positive_for_outlier_above_mean() -> None:
    context = build_corpus_context([80.0, 80.0, 80.0, 80.0, 95.0])
    assert z_score(95.0, context) > 1.0


def test_z_score_zero_std_returns_zero() -> None:
    context = build_corpus_context([80.0, 80.0, 80.0])
    assert z_score(80.0, context) == 0.0
    assert z_score(50.0, context) == 0.0


def test_percentile_rank_top_score_is_hundred() -> None:
    context = build_corpus_context([10.0, 20.0, 30.0, 40.0])
    assert percentile_rank(40.0, context) == pytest.approx(100.0)


def test_percentile_rank_bottom_score_is_lowest() -> None:
    context = build_corpus_context([10.0, 20.0, 30.0, 40.0])
    assert percentile_rank(10.0, context) == pytest.approx(25.0)


def test_percentile_rank_empty_context_is_zero() -> None:
    context = build_corpus_context([])
    assert percentile_rank(50.0, context) == 0.0
