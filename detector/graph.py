from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import networkx as nx

from config import Config
from detector.reader import (
    ArcEntity,
    CadDocument,
    CadEntity,
    CircleEntity,
    LineEntity,
    Point2D,
    PolylineEntity,
)


_EPSILON = 1e-9


@dataclass(frozen=True)
class GraphMetrics:
    degree_centrality: Tuple[Tuple[int, float], ...]
    betweenness_centrality: Tuple[Tuple[int, float], ...]
    clustering_coefficient: Tuple[Tuple[int, float], ...]
    component_count: int
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class _LinearSegment:
    entity_index: int
    start: Point2D
    end: Point2D


@dataclass(frozen=True)
class _CircularPrimitive:
    entity_index: int
    center: Point2D
    radius: float


def _distance2d(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_segments(
    points: Sequence[Point2D], closed: bool
) -> List[Tuple[Point2D, Point2D]]:
    count = len(points)
    if count < 2:
        return []
    segment_count = count if closed else count - 1
    return [(points[i], points[(i + 1) % count]) for i in range(segment_count)]


def _segment_angle(start: Point2D, end: Point2D) -> Optional[float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < _EPSILON and abs(dy) < _EPSILON:
        return None
    return math.atan2(dy, dx) % math.pi


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % math.pi
    return min(diff, math.pi - diff)


def _angles_parallel(a: float, b: float, tolerance: float) -> bool:
    return _angle_difference(a, b) <= tolerance


def _angles_perpendicular(a: float, b: float, tolerance: float) -> bool:
    return abs(_angle_difference(a, b) - math.pi / 2.0) <= tolerance


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point2D, b: Point2D, c: Point2D, tolerance: float) -> bool:
    if abs(_orientation(a, b, c)) > tolerance:
        return False
    return (
        min(a[0], b[0]) - tolerance <= c[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= c[1] <= max(a[1], b[1]) + tolerance
    )


def _segments_intersect(
    p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D, tolerance: float
) -> bool:
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    if (
        (d1 > tolerance and d2 < -tolerance) or (d1 < -tolerance and d2 > tolerance)
    ) and (
        (d3 > tolerance and d4 < -tolerance) or (d3 < -tolerance and d4 > tolerance)
    ):
        return True
    if _on_segment(p3, p4, p1, tolerance):
        return True
    if _on_segment(p3, p4, p2, tolerance):
        return True
    if _on_segment(p1, p2, p3, tolerance):
        return True
    if _on_segment(p1, p2, p4, tolerance):
        return True
    return False


def _point_segment_distance(point: Point2D, seg_start: Point2D, seg_end: Point2D) -> float:
    px, py = point
    ax, ay = seg_start
    bx, by = seg_end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared < _EPSILON:
        return _distance2d(point, seg_start)
    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t_clamped = max(0.0, min(1.0, t))
    closest = (ax + t_clamped * dx, ay + t_clamped * dy)
    return _distance2d(point, closest)


def _point_line_distance(point: Point2D, line_start: Point2D, line_end: Point2D) -> float:
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < _EPSILON:
        return _distance2d(point, line_start)
    return abs(dx * (ay - py) - (ax - px) * dy) / length


class GraphBuilder:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config if config is not None else Config()

    def build(self, document: CadDocument) -> nx.Graph:
        entities = document.entities
        graph = nx.Graph()
        for index, entity in enumerate(entities):
            graph.add_node(
                index,
                handle=entity.handle,
                entity_type=type(entity).__name__,
            )
        linear_segments = self._linear_segments(entities)
        circular_primitives = self._circular_primitives(entities)
        self._add_linear_relations(graph, linear_segments)
        self._add_circular_relations(graph, circular_primitives)
        self._add_mixed_relations(graph, linear_segments, circular_primitives)
        return graph

    def compute_metrics(self, graph: nx.Graph) -> GraphMetrics:
        degree = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)
        clustering = nx.clustering(graph)
        return GraphMetrics(
            degree_centrality=tuple(sorted(degree.items())),
            betweenness_centrality=tuple(sorted(betweenness.items())),
            clustering_coefficient=tuple(sorted(clustering.items())),
            component_count=nx.number_connected_components(graph),
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
        )

    def _linear_segments(self, entities: Sequence[CadEntity]) -> List[_LinearSegment]:
        segments: List[_LinearSegment] = []
        for index, entity in enumerate(entities):
            if isinstance(entity, LineEntity):
                segments.append(
                    _LinearSegment(
                        index,
                        (entity.start[0], entity.start[1]),
                        (entity.end[0], entity.end[1]),
                    )
                )
            elif isinstance(entity, PolylineEntity):
                for start, end in _polyline_segments(entity.points, entity.closed):
                    segments.append(_LinearSegment(index, start, end))
        return segments

    def _circular_primitives(
        self, entities: Sequence[CadEntity]
    ) -> List[_CircularPrimitive]:
        primitives: List[_CircularPrimitive] = []
        for index, entity in enumerate(entities):
            if isinstance(entity, ArcEntity):
                primitives.append(
                    _CircularPrimitive(
                        index, (entity.center[0], entity.center[1]), entity.radius
                    )
                )
            elif isinstance(entity, CircleEntity):
                primitives.append(
                    _CircularPrimitive(
                        index, (entity.center[0], entity.center[1]), entity.radius
                    )
                )
        return primitives

    def _add_linear_relations(
        self, graph: nx.Graph, segments: Sequence[_LinearSegment]
    ) -> None:
        intersection_tolerance = self._config.graph_intersection_tolerance
        parallel_tolerance = math.radians(self._config.graph_parallel_tolerance_degrees)
        perpendicular_tolerance = math.radians(
            self._config.graph_perpendicular_tolerance_degrees
        )
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                a, b = segments[i], segments[j]
                if a.entity_index == b.entity_index:
                    continue
                relations: List[str] = []
                if _segments_intersect(a.start, a.end, b.start, b.end, intersection_tolerance):
                    relations.append("intersection")
                angle_a = _segment_angle(a.start, a.end)
                angle_b = _segment_angle(b.start, b.end)
                if angle_a is not None and angle_b is not None:
                    if _angles_parallel(angle_a, angle_b, parallel_tolerance):
                        relations.append("parallel")
                    elif _angles_perpendicular(angle_a, angle_b, perpendicular_tolerance):
                        relations.append("perpendicular")
                if relations:
                    self._add_relations(graph, a.entity_index, b.entity_index, relations)

    def _add_circular_relations(
        self, graph: nx.Graph, primitives: Sequence[_CircularPrimitive]
    ) -> None:
        tangency_tolerance = self._config.graph_tangency_tolerance
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                a, b = primitives[i], primitives[j]
                if a.entity_index == b.entity_index:
                    continue
                distance = _distance2d(a.center, b.center)
                relations: List[str] = []
                if (
                    abs(distance - (a.radius + b.radius)) <= tangency_tolerance
                    or abs(distance - abs(a.radius - b.radius)) <= tangency_tolerance
                ):
                    relations.append("tangency")
                elif distance < a.radius + b.radius and distance > abs(a.radius - b.radius):
                    relations.append("intersection")
                if relations:
                    self._add_relations(graph, a.entity_index, b.entity_index, relations)

    def _add_mixed_relations(
        self,
        graph: nx.Graph,
        segments: Sequence[_LinearSegment],
        primitives: Sequence[_CircularPrimitive],
    ) -> None:
        tangency_tolerance = self._config.graph_tangency_tolerance
        for segment in segments:
            for primitive in primitives:
                if segment.entity_index == primitive.entity_index:
                    continue
                line_distance = _point_line_distance(
                    primitive.center, segment.start, segment.end
                )
                segment_distance = _point_segment_distance(
                    primitive.center, segment.start, segment.end
                )
                relations: List[str] = []
                if abs(line_distance - primitive.radius) <= tangency_tolerance:
                    relations.append("tangency")
                elif segment_distance <= primitive.radius:
                    relations.append("intersection")
                if relations:
                    self._add_relations(
                        graph, segment.entity_index, primitive.entity_index, relations
                    )

    def _add_relations(
        self, graph: nx.Graph, u: int, v: int, relations: Sequence[str]
    ) -> None:
        if graph.has_edge(u, v):
            existing = set(graph.edges[u, v]["relations"])
            existing.update(relations)
            graph.edges[u, v]["relations"] = tuple(sorted(existing))
        else:
            graph.add_edge(u, v, relations=tuple(sorted(set(relations))))
