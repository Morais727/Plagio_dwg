from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

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
_IDENTITY_ROTATION: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (1.0, 0.0),
    (0.0, 1.0),
)


@dataclass(frozen=True)
class NormalizationTransform:
    translation: Point3D
    rotation_matrix: Tuple[Tuple[float, float], Tuple[float, float]]
    rotation_angle_degrees: float
    scale: float

    def transform_point3d(self, point: Point3D) -> Point3D:
        x, y, z = point
        tx, ty, tz = self.translation
        cx, cy = x - tx, y - ty
        r00, r01 = self.rotation_matrix[0]
        r10, r11 = self.rotation_matrix[1]
        rx = r00 * cx + r01 * cy
        ry = r10 * cx + r11 * cy
        return (self.scale * rx, self.scale * ry, self.scale * (z - tz))

    def transform_point2d(self, point: Point2D) -> Point2D:
        x, y, _ = self.transform_point3d((point[0], point[1], 0.0))
        return (x, y)

    def transform_length(self, value: float) -> float:
        return value * self.scale

    def transform_angle_degrees(self, angle: float) -> float:
        return (angle + self.rotation_angle_degrees) % 360.0


class Normalizer:
    def normalize(self, document: CadDocument) -> CadDocument:
        transform = self.compute_transform(document)
        entities = tuple(
            self._transform_entity(entity, transform) for entity in document.entities
        )
        return replace(document, entities=entities)

    def compute_transform(self, document: CadDocument) -> NormalizationTransform:
        points = self._collect_points(document.entities)
        if len(points) < 2:
            return NormalizationTransform(
                translation=(0.0, 0.0, 0.0),
                rotation_matrix=_IDENTITY_ROTATION,
                rotation_angle_degrees=0.0,
                scale=1.0,
            )
        array = np.array(points, dtype=np.float64)
        center = array.mean(axis=0)
        centered_xy = array[:, :2] - center[:2]
        rotation_matrix = self._principal_rotation_matrix(centered_xy)
        rotated_xy = centered_xy @ rotation_matrix.T
        width = float(rotated_xy[:, 0].max() - rotated_xy[:, 0].min())
        height = float(rotated_xy[:, 1].max() - rotated_xy[:, 1].min())
        extent = max(width, height)
        scale = 1.0 / extent if extent > _EPSILON else 1.0
        angle_degrees = float(
            np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        )
        return NormalizationTransform(
            translation=(float(center[0]), float(center[1]), float(center[2])),
            rotation_matrix=(
                (float(rotation_matrix[0, 0]), float(rotation_matrix[0, 1])),
                (float(rotation_matrix[1, 0]), float(rotation_matrix[1, 1])),
            ),
            rotation_angle_degrees=angle_degrees,
            scale=scale,
        )

    def _collect_points(self, entities: Sequence[CadEntity]) -> List[Point3D]:
        points: List[Point3D] = []
        for entity in entities:
            points.extend(self._entity_points(entity))
        return points

    def _entity_points(self, entity: CadEntity) -> List[Point3D]:
        if isinstance(entity, LineEntity):
            return [entity.start, entity.end]
        if isinstance(entity, ArcEntity):
            return self._circle_bounding_points(entity.center, entity.radius)
        if isinstance(entity, CircleEntity):
            return self._circle_bounding_points(entity.center, entity.radius)
        if isinstance(entity, PolylineEntity):
            return [(x, y, 0.0) for x, y in entity.points]
        if isinstance(entity, InsertEntity):
            return [entity.insert_point]
        if isinstance(entity, TextEntity):
            return [entity.insert_point]
        if isinstance(entity, MTextEntity):
            return [entity.insert_point]
        return []

    def _circle_bounding_points(
        self, center: Point3D, radius: float
    ) -> List[Point3D]:
        cx, cy, cz = center
        return [
            (cx + radius, cy, cz),
            (cx - radius, cy, cz),
            (cx, cy + radius, cz),
            (cx, cy - radius, cz),
        ]

    def _principal_rotation_matrix(
        self, centered_xy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        covariance = np.cov(centered_xy, rowvar=False)
        if covariance.shape != (2, 2) or np.allclose(covariance, 0.0):
            return np.eye(2, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 1] *= -1.0
        projected = centered_xy @ eigenvectors
        skew = float(np.sum(projected[:, 0] ** 3))
        if skew < 0:
            eigenvectors = -eigenvectors
        return eigenvectors.T

    def _transform_entity(
        self, entity: CadEntity, transform: NormalizationTransform
    ) -> CadEntity:
        if isinstance(entity, LineEntity):
            return replace(
                entity,
                start=transform.transform_point3d(entity.start),
                end=transform.transform_point3d(entity.end),
            )
        if isinstance(entity, ArcEntity):
            return replace(
                entity,
                center=transform.transform_point3d(entity.center),
                radius=transform.transform_length(entity.radius),
                start_angle=transform.transform_angle_degrees(entity.start_angle),
                end_angle=transform.transform_angle_degrees(entity.end_angle),
            )
        if isinstance(entity, CircleEntity):
            return replace(
                entity,
                center=transform.transform_point3d(entity.center),
                radius=transform.transform_length(entity.radius),
            )
        if isinstance(entity, PolylineEntity):
            return replace(
                entity,
                points=tuple(
                    transform.transform_point2d(point) for point in entity.points
                ),
            )
        if isinstance(entity, InsertEntity):
            return replace(
                entity,
                insert_point=transform.transform_point3d(entity.insert_point),
                x_scale=transform.transform_length(entity.x_scale),
                y_scale=transform.transform_length(entity.y_scale),
                z_scale=transform.transform_length(entity.z_scale),
                rotation=transform.transform_angle_degrees(entity.rotation),
            )
        if isinstance(entity, TextEntity):
            return replace(
                entity,
                insert_point=transform.transform_point3d(entity.insert_point),
                height=transform.transform_length(entity.height),
            )
        if isinstance(entity, MTextEntity):
            return replace(
                entity,
                insert_point=transform.transform_point3d(entity.insert_point),
                char_height=transform.transform_length(entity.char_height),
            )
        if isinstance(entity, DimensionEntity):
            measurement = entity.measurement
            return replace(
                entity,
                insert_point=transform.transform_point3d(entity.insert_point),
                measurement=(
                    transform.transform_length(measurement)
                    if measurement is not None
                    else None
                ),
            )
        return entity
