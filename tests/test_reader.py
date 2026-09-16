from datetime import datetime
from pathlib import Path

import ezdxf
import pytest
from ezdxf.document import Drawing

from detector.reader import (
    ArcEntity,
    CadDocument,
    CadReader,
    CircleEntity,
    DimensionEntity,
    DocumentMetadata,
    DxfLoader,
    DxfReader,
    InsertEntity,
    LineEntity,
    MTextEntity,
    PolylineEntity,
    TextEntity,
    julian_to_datetime,
)


@pytest.fixture
def reader() -> DxfReader:
    return DxfReader()


@pytest.fixture
def sample_document() -> Drawing:
    document = ezdxf.new("R2010")
    document.layers.add("GEOMETRY")
    document.layers.add("BLOCKS")
    document.layers.add("TEXT")
    document.layers.add("DIMENSIONS")
    modelspace = document.modelspace()
    modelspace.add_line((0, 0), (1, 1), dxfattribs={"layer": "GEOMETRY"})
    modelspace.add_arc(
        (0, 0), 5, 0, 90, dxfattribs={"layer": "GEOMETRY"}
    )
    modelspace.add_circle((2, 2), 3, dxfattribs={"layer": "GEOMETRY"})
    modelspace.add_lwpolyline(
        [(0, 0), (1, 0), (1, 1)],
        close=True,
        dxfattribs={"layer": "GEOMETRY"},
    )
    block = document.blocks.new(name="MYBLOCK")
    block.add_line((0, 0), (1, 0))
    modelspace.add_blockref(
        "MYBLOCK",
        (5, 5),
        dxfattribs={
            "xscale": 1.0,
            "yscale": 1.0,
            "zscale": 1.0,
            "rotation": 45.0,
            "layer": "BLOCKS",
        },
    )
    modelspace.add_text(
        "hello", dxfattribs={"height": 2.5, "insert": (0, 0), "layer": "TEXT"}
    )
    modelspace.add_mtext(
        "hello\\Pworld",
        dxfattribs={"insert": (0, 0), "char_height": 1.5, "layer": "TEXT"},
    )
    dimension = modelspace.add_linear_dim(
        base=(0, 2), p1=(0, 0), p2=(1, 0), dxfattribs={"layer": "DIMENSIONS"}
    )
    dimension.render()
    modelspace.add_point((9, 9))
    return document


def test_read_document_extracts_all_supported_entity_types(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    entity_types = {type(entity) for entity in cad_document.entities}
    assert entity_types == {
        LineEntity,
        ArcEntity,
        CircleEntity,
        PolylineEntity,
        InsertEntity,
        TextEntity,
        MTextEntity,
        DimensionEntity,
    }


def test_unsupported_entity_types_are_ignored(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert len(cad_document.entities) == 8


def test_line_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    line = next(
        entity for entity in cad_document.entities if isinstance(entity, LineEntity)
    )
    assert line.start == (0.0, 0.0, 0.0)
    assert line.end == (1.0, 1.0, 0.0)
    assert isinstance(line.handle, str) and line.handle

def test_arc_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    arc = next(
        entity for entity in cad_document.entities if isinstance(entity, ArcEntity)
    )
    assert arc.center == (0.0, 0.0, 0.0)
    assert arc.radius == 5.0
    assert arc.start_angle == 0.0
    assert arc.end_angle == 90.0


def test_circle_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    circle = next(
        entity for entity in cad_document.entities if isinstance(entity, CircleEntity)
    )
    assert circle.center == (2.0, 2.0, 0.0)
    assert circle.radius == 3.0


def test_polyline_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    polyline = next(
        entity
        for entity in cad_document.entities
        if isinstance(entity, PolylineEntity)
    )
    assert polyline.points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert polyline.closed is True


def test_insert_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    insert = next(
        entity for entity in cad_document.entities if isinstance(entity, InsertEntity)
    )
    assert insert.block_name == "MYBLOCK"
    assert insert.insert_point == (5.0, 5.0, 0.0)
    assert insert.rotation == 45.0


def test_text_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    text = next(
        entity for entity in cad_document.entities if isinstance(entity, TextEntity)
    )
    assert text.text == "hello"
    assert text.height == 2.5
    assert text.style == "Standard"


def test_mtext_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    mtext = next(
        entity for entity in cad_document.entities if isinstance(entity, MTextEntity)
    )
    assert mtext.text == "hello\nworld"
    assert mtext.char_height == 1.5


def test_dimension_entity_extraction(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    dimension = next(
        entity
        for entity in cad_document.entities
        if isinstance(entity, DimensionEntity)
    )
    assert dimension.style == "Standard"
    assert dimension.measurement == pytest.approx(1.0)


def test_blocks_exclude_internal_model_and_paper_space(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert "MYBLOCK" in cad_document.blocks
    assert all(not name.startswith("*") for name in cad_document.blocks)


def test_text_and_dimension_styles_are_extracted(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert "Standard" in cad_document.text_styles
    assert "Standard" in cad_document.dimension_styles


def test_metadata_dxf_version_and_last_saved_by(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert cad_document.metadata.dxf_version == "AC1024"
    assert cad_document.metadata.last_saved_by == "ezdxf"
    assert cad_document.metadata.author == "ezdxf"


def test_metadata_author_prefers_custom_property(
    reader: DxfReader, sample_document: Drawing
) -> None:
    sample_document.header.custom_vars.append("Author", "Maria Silva")
    cad_document = reader.read_document(sample_document)
    assert cad_document.metadata.author == "Maria Silva"


def test_metadata_timestamps_are_parsed(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert isinstance(cad_document.metadata.created, datetime)
    assert isinstance(cad_document.metadata.modified, datetime)


def test_metadata_timestamps_absent_when_zero(
    reader: DxfReader, sample_document: Drawing
) -> None:
    sample_document.header["$TDCREATE"] = 0.0
    sample_document.header["$TDUPDATE"] = 0.0
    cad_document = reader.read_document(sample_document)
    assert cad_document.metadata.created is None
    assert cad_document.metadata.modified is None


def test_source_path_is_none_when_not_provided(
    reader: DxfReader, sample_document: Drawing
) -> None:
    cad_document = reader.read_document(sample_document)
    assert cad_document.source_path is None


def test_source_path_is_preserved_when_provided(
    reader: DxfReader, sample_document: Drawing, tmp_path: Path
) -> None:
    path = tmp_path / "drawing.dxf"
    cad_document = reader.read_document(sample_document, source_path=path)
    assert cad_document.source_path == path


def test_read_loads_from_disk_via_default_loader(
    reader: DxfReader, sample_document: Drawing, tmp_path: Path
) -> None:
    path = tmp_path / "drawing.dxf"
    sample_document.saveas(str(path))
    cad_document = reader.read(path)
    assert isinstance(cad_document, CadDocument)
    assert cad_document.source_path == path
    assert len(cad_document.entities) == 8


def test_read_uses_injected_loader(sample_document: Drawing, tmp_path: Path) -> None:
    class FakeLoader(DxfLoader):
        def __init__(self, document: Drawing) -> None:
            self._document = document

        def load(self, path: Path) -> Drawing:
            return self._document

    fake_loader = FakeLoader(sample_document)
    reader_with_fake_loader = DxfReader(loader=fake_loader)
    path = tmp_path / "does_not_need_to_exist.dxf"
    cad_document = reader_with_fake_loader.read(path)
    assert cad_document.source_path == path
    assert len(cad_document.entities) == 8


def test_julian_to_datetime_matches_j2000_reference() -> None:
    assert julian_to_datetime(2451545.0) == datetime(2000, 1, 1, 12, 0, 0)


def test_julian_to_datetime_matches_known_gregorian_date() -> None:
    assert julian_to_datetime(2299160.5) == datetime(1582, 10, 15, 0, 0, 0)


def test_julian_to_datetime_matches_known_julian_calendar_date() -> None:
    assert julian_to_datetime(2299159.5) == datetime(1582, 10, 4, 0, 0, 0)


def test_dimension_measurement_is_none_when_unavailable(reader: DxfReader) -> None:
    class BrokenDimensionDxf:
        handle = "FF"
        layer = "0"
        dimtype = 32
        dimstyle = "Standard"
        text = "<>"

        def __init__(self) -> None:
            self.defpoint = (0.0, 0.0, 0.0)

    class BrokenDimensionEntity:
        dxf = BrokenDimensionDxf()

        def get_measurement(self) -> float:
            raise ValueError("no geometric association")

    result = reader._parse_dimension(BrokenDimensionEntity())
    assert result.measurement is None


def test_cad_reader_uses_dxf_reader_for_dxf(
    sample_document: Drawing, tmp_path: Path
) -> None:
    path = tmp_path / "test.dxf"
    sample_document.saveas(str(path))
    reader = CadReader()
    cad_doc = reader.read(path)
    assert len(cad_doc.entities) == 8


def test_cad_reader_raises_on_unsupported_format() -> None:
    reader = CadReader()
    with pytest.raises(ValueError, match="Formato nao suportado"):
        reader.read(Path("test.pdf"))


def test_cad_reader_converts_dwg_to_dxf_and_reads(
    tmp_path: Path, mocker
) -> None:
    fake_cad_doc = CadDocument(
        source_path=tmp_path / "test.dxf",
        entities=(),
        blocks=(),
        text_styles=(),
        dimension_styles=(),
        metadata=DocumentMetadata(
            author=None, dxf_version="AC1024",
            last_saved_by=None, created=None, modified=None,
        ),
    )

    mock_converter = mocker.patch(
        "detector.dwg_converter.DwgConverter", autospec=True
    )
    mock_converter_instance = mock_converter.return_value
    fake_dxf_path = tmp_path / "converted.dxf"
    mock_converter_instance.convert.return_value = fake_dxf_path

    mock_dxf_reader = mocker.Mock(spec=DxfReader)
    mock_dxf_reader.read.return_value = fake_cad_doc

    reader = CadReader(dxf_reader=mock_dxf_reader)
    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")

    result = reader.read(dwg_path)

    mock_converter_instance.convert.assert_called_once_with(dwg_path)
    mock_dxf_reader.read.assert_called_once_with(fake_dxf_path)
    assert result.source_path == fake_cad_doc.source_path



