from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class CorpusContext:
    scores: Tuple[float, ...]
    mean: float
    std: float


def build_corpus_context(scores: Sequence[float]) -> CorpusContext:
    values = tuple(scores)
    count = len(values)
    mean = sum(values) / count if count else 0.0
    variance = sum((value - mean) ** 2 for value in values) / count if count else 0.0
    return CorpusContext(scores=values, mean=mean, std=math.sqrt(variance))


def z_score(value: float, context: CorpusContext) -> float:
    if context.std <= 1e-9:
        return 0.0
    return (value - context.mean) / context.std


def percentile_rank(value: float, context: CorpusContext) -> float:
    if not context.scores:
        return 0.0
    below_or_equal = sum(1 for score in context.scores if score <= value)
    return 100.0 * below_or_equal / len(context.scores)
