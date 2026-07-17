from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import ezdxf
from ezdxf.document import Drawing
from ezdxf.entities import DXFEntity
from ezdxf.math import Vec3


Point3D = tuple[float, float, float]
Point2D = tuple[float, float]


@dataclass(frozen=True)
class LineEntity:
    handle: str
    layer: str
    start: Point3D
    end: Point3D


@dataclass(frozen=True)
class ArcEntity:
    handle: str
    layer: str
    center: Point3D
    radius: float
    start_angle: float
    end_angle: float


@dataclass(frozen=True)
class CircleEntity:
    handle: str
    layer: str
    center: Point3D
    radius: float


@dataclass(frozen=True)
class PolylineEntity:
    handle: str
    layer: str
    points: tuple[Point2D, ...]
    closed: bool


@dataclass(frozen=True)
class InsertEntity:
    handle: str
    layer: str
    block_name: str
    insert_point: Point3D
    x_scale: float
    y_scale: float
    z_scale: float
    rotation: float


@dataclass(frozen=True)
class TextEntity:
    handle: str
    layer: str
    text: str
    insert_point: Point3D
    height: float
    style: str


@dataclass(frozen=True)
class MTextEntity:
    handle: str
    layer: str
    text: str
    insert_point: Point3D
    char_height: float
    style: str


@dataclass(frozen=True)
class DimensionEntity:
    handle: str
    layer: str
    dim_type: int
    style: str
    text_override: str
    measurement: Optional[float]


CadEntity = Union[
    LineEntity,
    ArcEntity,
    CircleEntity,
    PolylineEntity,
    InsertEntity,
    TextEntity,
    MTextEntity,
    DimensionEntity,
]


@dataclass(frozen=True)
class DocumentMetadata:
    author: Optional[str]
    dxf_version: str
    last_saved_by: Optional[str]
    created: Optional[datetime]
    modified: Optional[datetime]


@dataclass(frozen=True)
class CadDocument:
    source_path: Optional[Path]
    entities: tuple[CadEntity, ...]
    layers: tuple[str, ...]
    blocks: tuple[str, ...]
    text_styles: tuple[str, ...]
    dimension_styles: tuple[str, ...]
    metadata: DocumentMetadata


class DxfLoader:
    def load(self, path: Path) -> Drawing:
        return ezdxf.readfile(str(path))


def julian_to_datetime(julian_date: float) -> datetime:
    adjusted = julian_date + 0.5
    z = int(adjusted)
    f = adjusted - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    day_int = int(day)
    seconds = round((day - day_int) * 86400)
    return datetime(year, month, day_int) + timedelta(seconds=seconds)


def vec_to_point3d(vec: Vec3) -> Point3D:
    return (float(vec.x), float(vec.y), float(vec.z))


class DxfReader:
    def __init__(self, loader: Optional[DxfLoader] = None) -> None:
        self._loader = loader if loader is not None else DxfLoader()

    def read(self, path: Union[str, Path]) -> CadDocument:
        file_path = Path(path)
        document = self._loader.load(file_path)
        return self.read_document(document, file_path)

    def read_document(
        self, document: Drawing, source_path: Optional[Path] = None
    ) -> CadDocument:
        modelspace = document.modelspace()
        entities = tuple(
            entity
            for entity in (self._parse_entity(raw) for raw in modelspace)
            if entity is not None
        )
        return CadDocument(
            source_path=source_path,
            entities=entities,
            layers=self._extract_layers(document),
            blocks=self._extract_blocks(document),
            text_styles=self._extract_text_styles(document),
            dimension_styles=self._extract_dimension_styles(document),
            metadata=self._extract_metadata(document),
        )

    def _parse_entity(self, entity: DXFEntity) -> Optional[CadEntity]:
        dxftype = entity.dxftype()
        if dxftype == "LINE":
            return self._parse_line(entity)
        if dxftype == "ARC":
            return self._parse_arc(entity)
        if dxftype == "CIRCLE":
            return self._parse_circle(entity)
        if dxftype == "LWPOLYLINE":
            return self._parse_polyline(entity)
        if dxftype == "INSERT":
            return self._parse_insert(entity)
        if dxftype == "TEXT":
            return self._parse_text(entity)
        if dxftype == "MTEXT":
            return self._parse_mtext(entity)
        if dxftype == "DIMENSION":
            return self._parse_dimension(entity)
        return None

    def _parse_line(self, entity: DXFEntity) -> LineEntity:
        return LineEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            start=vec_to_point3d(Vec3(entity.dxf.start)),
            end=vec_to_point3d(Vec3(entity.dxf.end)),
        )

    def _parse_arc(self, entity: DXFEntity) -> ArcEntity:
        return ArcEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            center=vec_to_point3d(Vec3(entity.dxf.center)),
            radius=float(entity.dxf.radius),
            start_angle=float(entity.dxf.start_angle),
            end_angle=float(entity.dxf.end_angle),
        )

    def _parse_circle(self, entity: DXFEntity) -> CircleEntity:
        return CircleEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            center=vec_to_point3d(Vec3(entity.dxf.center)),
            radius=float(entity.dxf.radius),
        )

    def _parse_polyline(self, entity: DXFEntity) -> PolylineEntity:
        points = tuple(
            (float(x), float(y)) for x, y in entity.get_points("xy")
        )
        return PolylineEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            points=points,
            closed=bool(entity.closed),
        )

    def _parse_insert(self, entity: DXFEntity) -> InsertEntity:
        return InsertEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            block_name=entity.dxf.name,
            insert_point=vec_to_point3d(Vec3(entity.dxf.insert)),
            x_scale=float(entity.dxf.xscale),
            y_scale=float(entity.dxf.yscale),
            z_scale=float(entity.dxf.zscale),
            rotation=float(entity.dxf.rotation),
        )

    def _parse_text(self, entity: DXFEntity) -> TextEntity:
        return TextEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            text=entity.dxf.text,
            insert_point=vec_to_point3d(Vec3(entity.dxf.insert)),
            height=float(entity.dxf.height),
            style=entity.dxf.style,
        )

    def _parse_mtext(self, entity: DXFEntity) -> MTextEntity:
        return MTextEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            text=entity.plain_text(),
            insert_point=vec_to_point3d(Vec3(entity.dxf.insert)),
            char_height=float(entity.dxf.char_height),
            style=entity.dxf.style,
        )

    def _parse_dimension(self, entity: DXFEntity) -> DimensionEntity:
        try:
            measurement: Optional[float] = float(entity.get_measurement())
        except Exception:
            measurement = None
        return DimensionEntity(
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            dim_type=int(entity.dxf.dimtype),
            style=entity.dxf.dimstyle,
            text_override=entity.dxf.text,
            measurement=measurement,
        )

    def _extract_layers(self, document: Drawing) -> tuple[str, ...]:
        return tuple(layer.dxf.name for layer in document.layers)

    def _extract_blocks(self, document: Drawing) -> tuple[str, ...]:
        return tuple(
            block.name for block in document.blocks if not block.name.startswith("*")
        )

    def _extract_text_styles(self, document: Drawing) -> tuple[str, ...]:
        return tuple(style.dxf.name for style in document.styles)

    def _extract_dimension_styles(self, document: Drawing) -> tuple[str, ...]:
        return tuple(dimstyle.dxf.name for dimstyle in document.dimstyles)

    def _extract_metadata(self, document: Drawing) -> DocumentMetadata:
        header = document.header
        last_saved_by = header.get("$LASTSAVEDBY", None)
        author = self._extract_author(document, last_saved_by)
        created = self._extract_timestamp(header, "$TDCREATE")
        modified = self._extract_timestamp(header, "$TDUPDATE")
        return DocumentMetadata(
            author=author,
            dxf_version=document.dxfversion,
            last_saved_by=last_saved_by,
            created=created,
            modified=modified,
        )

    def _extract_author(
        self, document: Drawing, last_saved_by: Optional[str]
    ) -> Optional[str]:
        for tag, value in document.header.custom_vars.properties:
            if tag.strip().lower() == "author":
                return value
        return last_saved_by

    def _extract_timestamp(self, header, variable_name: str) -> Optional[datetime]:
        value = header.get(variable_name, None)
        if value is None or float(value) <= 0:
            return None
        return julian_to_datetime(float(value))
