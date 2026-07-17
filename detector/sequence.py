from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Type

from rapidfuzz.distance import LCSseq, Levenshtein

from config import Config
from detector.reader import (
    ArcEntity,
    CadDocument,
    CadEntity,
    CircleEntity,
    DimensionEntity,
    InsertEntity,
    LineEntity,
    MTextEntity,
    PolylineEntity,
    TextEntity,
)


_ENTITY_CODES: Dict[Type[CadEntity], str] = {
    LineEntity: "L",
    ArcEntity: "A",
    CircleEntity: "C",
    PolylineEntity: "P",
    InsertEntity: "I",
    TextEntity: "T",
    MTextEntity: "M",
    DimensionEntity: "D",
}

_UNKNOWN_ENTITY_CODE = "?"


@dataclass(frozen=True)
class SequenceComparisonResult:
    sequence_a_length: int
    sequence_b_length: int
    lcs_length: int
    lcs_ratio: float
    levenshtein_distance: int
    levenshtein_ratio: float
    combined_score: float
    dtw_distance: Optional[float]


def entity_code(entity: CadEntity) -> str:
    return _ENTITY_CODES.get(type(entity), _UNKNOWN_ENTITY_CODE)


class SequenceAnalyzer:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config if config is not None else Config()

    def build_sequence(self, document: CadDocument) -> str:
        return "".join(entity_code(entity) for entity in document.entities)

    def compare(
        self, sequence_a: str, sequence_b: str, include_dtw: bool = False
    ) -> SequenceComparisonResult:
        lcs_length = LCSseq.similarity(sequence_a, sequence_b)
        lcs_ratio = LCSseq.normalized_similarity(sequence_a, sequence_b)
        levenshtein_distance = Levenshtein.distance(sequence_a, sequence_b)
        levenshtein_ratio = Levenshtein.normalized_similarity(sequence_a, sequence_b)
        combined_score = (
            self._config.lcs_weight * lcs_ratio
            + self._config.levenshtein_weight * levenshtein_ratio
        )
        dtw_distance = (
            self.dynamic_time_warping(sequence_a, sequence_b) if include_dtw else None
        )
        return SequenceComparisonResult(
            sequence_a_length=len(sequence_a),
            sequence_b_length=len(sequence_b),
            lcs_length=lcs_length,
            lcs_ratio=lcs_ratio,
            levenshtein_distance=levenshtein_distance,
            levenshtein_ratio=levenshtein_ratio,
            combined_score=combined_score,
            dtw_distance=dtw_distance,
        )

    def dynamic_time_warping(self, sequence_a: str, sequence_b: str) -> float:
        length_a = len(sequence_a)
        length_b = len(sequence_b)
        if length_a == 0 or length_b == 0:
            return float(max(length_a, length_b))
        previous_row = [float(j) for j in range(length_b + 1)]
        for i in range(1, length_a + 1):
            current_row = [float(i)] + [math.inf] * length_b
            for j in range(1, length_b + 1):
                cost = 0.0 if sequence_a[i - 1] == sequence_b[j - 1] else 1.0
                current_row[j] = cost + min(
                    previous_row[j], current_row[j - 1], previous_row[j - 1]
                )
            previous_row = current_row
        return previous_row[length_b]
