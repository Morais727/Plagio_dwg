import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytest

from config import Config
from detector.reader import (
    ArcEntity,
    CadDocument,
    CircleEntity,
    DimensionEntity,
    DocumentMetadata,
    InsertEntity,
    LineEntity,
    MTextEntity,
    PolylineEntity,
    TextEntity,
)
from detector.report import PairReport, ReportGenerator
from detector.similarity import ComponentScore, SimilarityResult


@pytest.fixture
def generator() -> ReportGenerator:
    return ReportGenerator()


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        author="Maria",
        dxf_version="AC1032",
        last_saved_by="Maria",
        created=datetime(2026, 1, 1),
        modified=datetime(2026, 1, 2),
    )


def _document(
    source_path: Optional[Path] = None,
    entities: Tuple = (),
) -> CadDocument:
    return CadDocument(
        source_path=source_path,
        entities=entities,
        layers=("0", "COTAS"),
        blocks=("BLOCO_A",),
        text_styles=("Standard",),
        dimension_styles=("Standard",),
        metadata=_metadata(),
    )


def _all_entity_types_document(source_path: Optional[Path] = None) -> CadDocument:
    entities = (
        LineEntity(handle="1", layer="0", start=(0.0, 0.0, 0.0), end=(10.0, 10.0, 0.0)),
        ArcEntity(
            handle="2", layer="0", center=(0.0, 0.0, 0.0), radius=5.0,
            start_angle=0.0, end_angle=90.0,
        ),
        CircleEntity(handle="3", layer="0", center=(5.0, 5.0, 0.0), radius=2.0),
        PolylineEntity(
            handle="4", layer="0",
            points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            closed=True,
        ),
        InsertEntity(
            handle="5", layer="0", block_name="BLOCO_A",
            insert_point=(2.0, 2.0, 0.0), x_scale=1.0, y_scale=1.0, z_scale=1.0,
            rotation=0.0,
        ),
        TextEntity(
            handle="6", layer="0", text="A", insert_point=(1.0, 1.0, 0.0),
            height=0.5, style="Standard",
        ),
        MTextEntity(
            handle="7", layer="0", text="B", insert_point=(3.0, 3.0, 0.0),
            char_height=0.5, style="Standard",
        ),
        DimensionEntity(
            handle="8", layer="0", dim_type=0, style="Standard",
            text_override="", measurement=10.0,
        ),
    )
    return _document(source_path=source_path, entities=entities)


def _similarity_result(total_score: float = 67.5) -> SimilarityResult:
    components = (
        ComponentScore("geometry", 0.9, 0.35, "Geometria: 90.0% de similaridade"),
        ComponentScore("sequence", 0.8, 0.25, "Sequência: 80.0% de similaridade"),
        ComponentScore("graph", 0.5, 0.20, "Grafo: 50.0% de similaridade"),
        ComponentScore("styles", 0.6, 0.10, "Estilos/Layers: 60.0% de similaridade"),
    )
    return SimilarityResult(total_score=total_score, component_scores=components)


def test_build_metrics_table_has_expected_columns_and_row_count(
    generator: ReportGenerator,
) -> None:
    table = generator.build_metrics_table(_similarity_result())
    assert list(table.columns) == ["componente", "peso", "contribuicao", "score"]
    assert len(table) == 4
    assert isinstance(table, pd.DataFrame)


def test_build_metrics_table_translates_component_names(generator: ReportGenerator) -> None:
    table = generator.build_metrics_table(_similarity_result())
    assert "Geometria" in table["componente"].values
    assert "Sequência" in table["componente"].values


def test_build_metrics_table_computes_contribution(generator: ReportGenerator) -> None:
    table = generator.build_metrics_table(_similarity_result())
    geometry_row = table[table["componente"] == "Geometria"].iloc[0]
    assert geometry_row["score"] == pytest.approx(90.0)
    assert geometry_row["contribuicao"] == pytest.approx(31.5)


def test_build_justification_mentions_total_score(generator: ReportGenerator) -> None:
    justification = generator.build_justification(_similarity_result(72.5))
    assert "72.5" in justification


def test_build_justification_includes_component_details(generator: ReportGenerator) -> None:
    justification = generator.build_justification(_similarity_result())
    assert "Geometria" in justification
    assert "Sequência" in justification


def test_build_justification_high_score_yields_high_suspicion_text(
    generator: ReportGenerator,
) -> None:
    justification = generator.build_justification(_similarity_result(90.0))
    assert "muito alta" in justification


def test_build_justification_low_score_yields_low_suspicion_text(
    generator: ReportGenerator,
) -> None:
    justification = generator.build_justification(_similarity_result(5.0))
    assert "Baixa suspeita" in justification


def test_render_comparison_image_returns_valid_png_bytes(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    image_bytes = generator.render_comparison_image(document, document, "A", "B")
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_comparison_image_handles_empty_documents(generator: ReportGenerator) -> None:
    document = _document()
    image_bytes = generator.render_comparison_image(document, document, "A", "B")
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_build_report_uses_source_path_stem_as_default_name(generator: ReportGenerator) -> None:
    document_a = _all_entity_types_document(source_path=Path("aluno1.dxf"))
    document_b = _all_entity_types_document(source_path=Path("aluno2.dxf"))
    report = generator.build_report(document_a, document_b, _similarity_result())
    assert report.document_a_name == "aluno1"
    assert report.document_b_name == "aluno2"


def test_build_report_uses_fallback_name_when_source_path_missing(
    generator: ReportGenerator,
) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    assert report.document_a_name == "Documento A"
    assert report.document_b_name == "Documento B"


def test_build_report_accepts_explicit_names(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(
        document, document, _similarity_result(), name_a="Turma1", name_b="Turma2"
    )
    assert report.document_a_name == "Turma1"
    assert report.document_b_name == "Turma2"


def test_build_report_returns_pair_report_with_expected_fields(
    generator: ReportGenerator,
) -> None:
    document = _all_entity_types_document()
    similarity_result = _similarity_result()
    report = generator.build_report(document, document, similarity_result)
    assert isinstance(report, PairReport)
    assert report.similarity_result is similarity_result
    assert isinstance(report.metrics_table, pd.DataFrame)
    assert report.comparison_image_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_pair_report_is_immutable(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    with pytest.raises(Exception):
        report.document_a_name = "outro"  # type: ignore[misc]


def test_generate_html_contains_score_and_image(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result(72.5))
    html = generator.generate_html(report)
    assert "72.5" in html
    assert "data:image/png;base64," in html
    assert "<table" in html


def test_generate_html_contains_document_names(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(
        document, document, _similarity_result(), name_a="AlunoX", name_b="AlunoY"
    )
    html = generator.generate_html(report)
    assert "AlunoX" in html
    assert "AlunoY" in html


def test_export_html_writes_file(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    tmp_path = Path(tempfile.mkdtemp())
    output_path = tmp_path / "nested" / "report.html"
    result_path = generator.export_html(report, output_path)
    assert result_path == output_path
    assert output_path.exists()
    assert "<!DOCTYPE html>" in output_path.read_text(encoding="utf-8")


def test_export_pdf_writes_valid_pdf_file(generator: ReportGenerator) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    tmp_path = Path(tempfile.mkdtemp())
    output_path = tmp_path / "nested" / "report.pdf"
    result_path = generator.export_pdf(report, output_path)
    assert result_path == output_path
    assert output_path.exists()
    with output_path.open("rb") as file_handle:
        header = file_handle.read(5)
    assert header == b"%PDF-"


def test_export_report_returns_html_and_pdf_paths(
    generator: ReportGenerator,
) -> None:
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    tmp_path = Path(tempfile.mkdtemp())
    html_path, pdf_path = generator.export_report(report, tmp_path, "par_1_2")
    assert html_path == tmp_path / "par_1_2.html"
    assert pdf_path == tmp_path / "par_1_2.pdf"
    assert html_path.exists()
    assert pdf_path.exists()


def test_export_report_defaults_to_config_output_dir() -> None:
    tmp_path = Path(tempfile.mkdtemp())
    config = Config(output_dir=tmp_path / "reports")
    generator = ReportGenerator(config=config)
    document = _all_entity_types_document()
    report = generator.build_report(document, document, _similarity_result())
    html_path, pdf_path = generator.export_report(report, None, "par_default")
    assert html_path.parent == tmp_path / "reports"
    assert pdf_path.parent == tmp_path / "reports"
    assert html_path.exists()
    assert pdf_path.exists()
