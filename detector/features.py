from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    Point2D,
    Point3D,
    PolylineEntity,
    TextEntity,
)


_EPSILON = 1e-9
_MAX_DECIMAL_PLACES = 6

_ENTITY_TYPE_NAMES: Dict[type, str] = {
    LineEntity: "LINE",
    ArcEntity: "ARC",
    CircleEntity: "CIRCLE",
    PolylineEntity: "LWPOLYLINE",
    InsertEntity: "INSERT",
    TextEntity: "TEXT",
    MTextEntity: "MTEXT",
    DimensionEntity: "DIMENSION",
}


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width: float
    height: float
    area: float


@dataclass(frozen=True)
class FeatureVector:
    entity_type_counts: Tuple[Tuple[str, int], ...]
    angle_histogram: Tuple[float, ...]
    length_histogram: Tuple[float, ...]
    bounding_box: BoundingBox
    spatial_density: Tuple[float, ...]
    layer_entity_counts: Tuple[Tuple[str, int], ...]
    block_usage_counts: Tuple[Tuple[str, int], ...]
    decimal_precision_histogram: Tuple[float, ...]


def _distance2d(a: Point2D, b: Point2D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _line_angle_degrees(start: Point2D, end: Point2D) -> Optional[float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < _EPSILON and abs(dy) < _EPSILON:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _polyline_segments(points: Tuple[Point2D, ...], closed: bool) -> List[Tuple[Point2D, Point2D]]:
    count = len(points)
    if count < 2:
        return []
    segment_count = count if closed else count - 1
    return [(points[i], points[(i + 1) % count]) for i in range(segment_count)]


def _circle_bounding_points(center: Point3D, radius: float) -> List[Point2D]:
    cx, cy, _ = center
    return [
        (cx + radius, cy),
        (cx - radius, cy),
        (cx, cy + radius),
        (cx, cy - radius),
    ]


def _representative_points(entity: CadEntity) -> List[Point2D]:
    if isinstance(entity, LineEntity):
        return [(entity.start[0], entity.start[1]), (entity.end[0], entity.end[1])]
    if isinstance(entity, ArcEntity):
        return _circle_bounding_points(entity.center, entity.radius)
    if isinstance(entity, CircleEntity):
        return _circle_bounding_points(entity.center, entity.radius)
    if isinstance(entity, PolylineEntity):
        return list(entity.points)
    if isinstance(entity, InsertEntity):
        return [(entity.insert_point[0], entity.insert_point[1])]
    if isinstance(entity, TextEntity):
        return [(entity.insert_point[0], entity.insert_point[1])]
    if isinstance(entity, MTextEntity):
        return [(entity.insert_point[0], entity.insert_point[1])]
    return []


def _angle_samples(entities: Sequence[CadEntity]) -> List[float]:
    samples: List[float] = []
    for entity in entities:
        if isinstance(entity, LineEntity):
            angle = _line_angle_degrees(
                (entity.start[0], entity.start[1]), (entity.end[0], entity.end[1])
            )
            if angle is not None:
                samples.append(angle)
        elif isinstance(entity, ArcEntity):
            samples.append(entity.start_angle % 360.0)
            samples.append(entity.end_angle % 360.0)
        elif isinstance(entity, PolylineEntity):
            for start, end in _polyline_segments(entity.points, entity.closed):
                angle = _line_angle_degrees(start, end)
                if angle is not None:
                    samples.append(angle)
        elif isinstance(entity, InsertEntity):
            samples.append(entity.rotation % 360.0)
    return samples


def _length_samples(entities: Sequence[CadEntity]) -> List[float]:
    samples: List[float] = []
    for entity in entities:
        if isinstance(entity, LineEntity):
            length = _distance2d(
                (entity.start[0], entity.start[1]), (entity.end[0], entity.end[1])
            )
            if length > _EPSILON:
                samples.append(length)
        elif isinstance(entity, ArcEntity):
            span = (entity.end_angle - entity.start_angle) % 360.0
            length = entity.radius * math.radians(span)
            if length > _EPSILON:
                samples.append(length)
        elif isinstance(entity, CircleEntity):
            length = 2.0 * math.pi * entity.radius
            if length > _EPSILON:
                samples.append(length)
        elif isinstance(entity, PolylineEntity):
            for start, end in _polyline_segments(entity.points, entity.closed):
                length = _distance2d(start, end)
                if length > _EPSILON:
                    samples.append(length)
        elif isinstance(entity, DimensionEntity):
            if entity.measurement is not None and entity.measurement > _EPSILON:
                samples.append(entity.measurement)
    return samples


def _numeric_values(entity: CadEntity) -> List[float]:
    if isinstance(entity, LineEntity):
        return [*entity.start, *entity.end]
    if isinstance(entity, ArcEntity):
        return [*entity.center, entity.radius]
    if isinstance(entity, CircleEntity):
        return [*entity.center, entity.radius]
    if isinstance(entity, PolylineEntity):
        values: List[float] = []
        for x, y in entity.points:
            values.extend([x, y])
        return values
    if isinstance(entity, InsertEntity):
        return [*entity.insert_point, entity.x_scale, entity.y_scale, entity.z_scale]
    if isinstance(entity, TextEntity):
        return [*entity.insert_point, entity.height]
    if isinstance(entity, MTextEntity):
        return [*entity.insert_point, entity.char_height]
    if isinstance(entity, DimensionEntity):
        return [entity.measurement] if entity.measurement is not None else []
    return []


def _decimal_places(value: float, max_places: int) -> int:
    for places in range(max_places + 1):
        if abs(round(value, places) - value) < 1e-9:
            return places
    return max_places


def _bin_index(value: float, minimum: float, span: float, bins: int) -> int:
    if span <= _EPSILON:
        return 0
    index = int((value - minimum) / span * bins)
    if index >= bins:
        return bins - 1
    if index < 0:
        return 0
    return index


def _histogram(
    values: Sequence[float], bins: int, minimum: float, maximum: float
) -> Tuple[float, ...]:
    counts = [0 for _ in range(bins)]
    span = maximum - minimum
    for value in values:
        counts[_bin_index(value, minimum, span, bins)] += 1
    total = float(sum(counts))
    if total <= 0.0:
        return tuple(0.0 for _ in range(bins))
    return tuple(count / total for count in counts)


class FeatureExtractor:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config if config is not None else Config()

    def extract(self, document: CadDocument) -> FeatureVector:
        entities = document.entities
        points = self._all_points(entities)
        length_samples = _length_samples(entities)
        length_maximum = max(length_samples) if length_samples else 1.0
        return FeatureVector(
            entity_type_counts=self._entity_type_counts(entities),
            angle_histogram=_histogram(
                _angle_samples(entities), self._config.angle_bins, 0.0, 360.0
            ),
            length_histogram=_histogram(
                length_samples, self._config.length_bins, 0.0, length_maximum
            ),
            bounding_box=self._bounding_box(points),
            spatial_density=self._spatial_density(points),
            layer_entity_counts=self._layer_entity_counts(entities),
            block_usage_counts=self._block_usage_counts(entities),
            decimal_precision_histogram=self._decimal_precision_histogram(entities),
        )

    def _all_points(self, entities: Sequence[CadEntity]) -> List[Point2D]:
        points: List[Point2D] = []
        for entity in entities:
            points.extend(_representative_points(entity))
        return points

    def _entity_type_counts(
        self, entities: Sequence[CadEntity]
    ) -> Tuple[Tuple[str, int], ...]:
        counts: Dict[str, int] = {name: 0 for name in _ENTITY_TYPE_NAMES.values()}
        for entity in entities:
            name = _ENTITY_TYPE_NAMES.get(type(entity))
            if name is not None:
                counts[name] += 1
        return tuple(sorted(counts.items()))

    def _bounding_box(self, points: Sequence[Point2D]) -> BoundingBox:
        if not points:
            return BoundingBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        return BoundingBox(min_x, min_y, max_x, max_y, width, height, width * height)

    def _spatial_density(self, points: Sequence[Point2D]) -> Tuple[float, ...]:
        grid = self._config.spatial_grid
        total_cells = grid * grid
        if not points:
            return tuple(0.0 for _ in range(total_cells))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        counts = [0 for _ in range(total_cells)]
        for x, y in points:
            col = _bin_index(x, min_x, width, grid)
            row = _bin_index(y, min_y, height, grid)
            counts[row * grid + col] += 1
        total = float(sum(counts))
        return tuple(count / total for count in counts)

    def _layer_entity_counts(
        self, entities: Sequence[CadEntity]
    ) -> Tuple[Tuple[str, int], ...]:
        counts: Dict[str, int] = {}
        for entity in entities:
            counts[entity.layer] = counts.get(entity.layer, 0) + 1
        return tuple(sorted(counts.items()))

    def _block_usage_counts(
        self, entities: Sequence[CadEntity]
    ) -> Tuple[Tuple[str, int], ...]:
        counts: Dict[str, int] = {}
        for entity in entities:
            if isinstance(entity, InsertEntity):
                counts[entity.block_name] = counts.get(entity.block_name, 0) + 1
        return tuple(sorted(counts.items()))

    def _decimal_precision_histogram(
        self, entities: Sequence[CadEntity]
    ) -> Tuple[float, ...]:
        counts = [0 for _ in range(_MAX_DECIMAL_PLACES + 1)]
        for entity in entities:
            for value in _numeric_values(entity):
                counts[_decimal_places(value, _MAX_DECIMAL_PLACES)] += 1
        total = float(sum(counts))
        if total <= 0.0:
            return tuple(0.0 for _ in counts)
        return tuple(count / total for count in counts)
