from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import ezdxf
import pytest
from ezdxf.document import Drawing

from config import Config
from detector.features import FeatureExtractor, FeatureVector
from detector.graph import GraphBuilder, GraphMetrics
from detector.normalizer import Normalizer
from detector.reader import CadDocument, DxfReader
from detector.report import PairReport, ReportGenerator
from detector.sequence import SequenceAnalyzer
from detector.similarity import SimilarityEngine, SimilarityResult


Point2D = Tuple[float, float]


def _transform_point(
    point: Point2D, angle_degrees: float, scale: float, dx: float, dy: float
) -> Point2D:
    angle = math.radians(angle_degrees)
    x, y = point
    rx = x * math.cos(angle) - y * math.sin(angle)
    ry = x * math.sin(angle) + y * math.cos(angle)
    return (rx * scale + dx, ry * scale + dy)


def _build_bracket_drawing(
    angle_degrees: float = 0.0,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    author: str = "Aluno A",
    layer_geometry: str = "GEOMETRIA",
    layer_cotas: str = "COTAS",
    block_name: str = "PARAFUSO",
    text_style: str = "Standard",
) -> Drawing:
    document = ezdxf.new("R2010")
    document.layers.add(layer_geometry)
    document.layers.add(layer_cotas)
    document.header.custom_vars.append("author", author)
    modelspace = document.modelspace()

    def t(point: Point2D) -> Point2D:
        return _transform_point(point, angle_degrees, scale, dx, dy)

    outline = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]
    for start, end in zip(outline, outline[1:] + outline[:1]):
        modelspace.add_line(t(start), t(end), dxfattribs={"layer": layer_geometry})

    modelspace.add_circle(
        t((5.0, 3.0)), radius=1.5 * scale, dxfattribs={"layer": layer_geometry}
    )
    modelspace.add_arc(
        t((2.0, 2.0)),
        radius=1.0 * scale,
        start_angle=(angle_degrees) % 360.0,
        end_angle=(90.0 + angle_degrees) % 360.0,
        dxfattribs={"layer": layer_geometry},
    )
    modelspace.add_lwpolyline(
        [t(p) for p in [(1.0, 1.0), (3.0, 1.0), (3.0, 2.0)]],
        dxfattribs={"layer": layer_geometry},
    )

    block = document.blocks.new(name=block_name)
    block.add_line((0.0, 0.0), (0.5, 0.0))
    modelspace.add_blockref(
        block_name,
        t((8.0, 5.0)),
        dxfattribs={
            "xscale": scale,
            "yscale": scale,
            "zscale": scale,
            "rotation": angle_degrees % 360.0,
            "layer": layer_geometry,
        },
    )

    modelspace.add_text(
        "PECA-01",
        dxfattribs={
            "height": 0.3 * scale,
            "insert": t((0.5, 6.5)),
            "layer": layer_cotas,
            "style": text_style,
        },
    )
    modelspace.add_mtext(
        "Escala 1:1",
        dxfattribs={
            "insert": t((0.5, 7.2)),
            "char_height": 0.25 * scale,
            "layer": layer_cotas,
            "style": text_style,
        },
    )
    dimension = modelspace.add_linear_dim(
        base=t((0.0, -1.0)),
        p1=t((0.0, 0.0)),
        p2=t((10.0, 0.0)),
        dxfattribs={"layer": layer_cotas},
    )
    dimension.render()

    return document


def _build_independent_drawing(author: str = "Aluno B") -> Drawing:
    document = ezdxf.new("R2010")
    document.layers.add("CONTORNO")
    document.layers.add("ANOTACOES")
    document.header.custom_vars.append("author", author)
    modelspace = document.modelspace()

    modelspace.add_circle((5.0, 3.0), radius=1.5, dxfattribs={"layer": "CONTORNO"})
    modelspace.add_lwpolyline(
        [(0.2, 0.2), (9.8, 0.2), (9.8, 5.8), (0.2, 5.8)],
        close=True,
        dxfattribs={"layer": "CONTORNO"},
    )
    modelspace.add_arc(
        (2.0, 2.0), 1.234, 15.0, 100.0, dxfattribs={"layer": "CONTORNO"}
    )
    modelspace.add_line(
        (1.0, 1.0), (3.5, 1.75), dxfattribs={"layer": "CONTORNO"}
    )
    modelspace.add_text(
        "Componente Alternativo",
        dxfattribs={
            "height": 0.4,
            "insert": (0.5, 6.5),
            "layer": "ANOTACOES",
        },
    )
    modelspace.add_mtext(
        "Projeto independente",
        dxfattribs={"insert": (0.5, 7.2), "char_height": 0.3, "layer": "ANOTACOES"},
    )

    return document


@pytest.fixture
def config() -> Config:
    return Config()


def _write_dxf(document: Drawing, path: Path) -> Path:
    document.saveas(str(path))
    return path


def _read(path: Path) -> CadDocument:
    return DxfReader().read(path)


def _run_pipeline(
    config: Config, document_a: CadDocument, document_b: CadDocument
) -> SimilarityResult:
    normalizer = Normalizer()
    normalized_a = normalizer.normalize(document_a)
    normalized_b = normalizer.normalize(document_b)

    extractor = FeatureExtractor(config)
    feature_a: FeatureVector = extractor.extract(normalized_a)
    feature_b: FeatureVector = extractor.extract(normalized_b)

    graph_builder = GraphBuilder(config)
    metrics_a: GraphMetrics = graph_builder.compute_metrics(
        graph_builder.build(normalized_a)
    )
    metrics_b: GraphMetrics = graph_builder.compute_metrics(
        graph_builder.build(normalized_b)
    )

    sequence_analyzer = SequenceAnalyzer(config)
    sequence_result = sequence_analyzer.compare(
        sequence_analyzer.build_sequence(normalized_a),
        sequence_analyzer.build_sequence(normalized_b),
    )

    engine = SimilarityEngine(config)
    return engine.compute_score(
        feature_a,
        feature_b,
        sequence_result,
        metrics_a,
        metrics_b,
        normalized_a,
        normalized_b,
    )


def test_pipeline_scores_transformed_copy_higher_than_independent_work(
    tmp_path: Path, config: Config
) -> None:
    original_path = _write_dxf(
        _build_bracket_drawing(author="Aluno A"), tmp_path / "original.dxf"
    )
    copy_path = _write_dxf(
        _build_bracket_drawing(
            angle_degrees=37.0, scale=2.5, dx=120.0, dy=-40.0, author="Aluno A"
        ),
        tmp_path / "copy.dxf",
    )
    independent_path = _write_dxf(
        _build_independent_drawing(author="Aluno B"), tmp_path / "independent.dxf"
    )

    original_document = _read(original_path)
    copy_document = _read(copy_path)
    independent_document = _read(independent_path)

    copy_result = _run_pipeline(config, original_document, copy_document)
    independent_result = _run_pipeline(config, original_document, independent_document)

    assert 0.0 <= copy_result.total_score <= 100.0
    assert 0.0 <= independent_result.total_score <= 100.0
    assert copy_result.total_score > independent_result.total_score
    assert copy_result.total_score >= 70.0


def test_pipeline_scores_identical_document_near_maximum(
    tmp_path: Path, config: Config
) -> None:
    path_a = _write_dxf(_build_bracket_drawing(), tmp_path / "a.dxf")
    path_b = _write_dxf(_build_bracket_drawing(), tmp_path / "b.dxf")

    result = _run_pipeline(config, _read(path_a), _read(path_b))

    assert result.total_score >= 95.0


def test_pipeline_component_scores_sum_to_total_with_configured_weights(
    tmp_path: Path, config: Config
) -> None:
    path_a = _write_dxf(_build_bracket_drawing(), tmp_path / "a.dxf")
    path_b = _write_dxf(
        _build_bracket_drawing(angle_degrees=90.0, scale=1.8, dx=5.0, dy=5.0),
        tmp_path / "b.dxf",
    )

    result = _run_pipeline(config, _read(path_a), _read(path_b))

    expected_total = (
        sum(component.score * component.weight for component in result.component_scores)
        * 100.0
    )
    assert result.total_score == pytest.approx(expected_total, abs=1e-6)
    assert {component.name for component in result.component_scores} == set(
        config.weights.keys()
    )


def test_full_pipeline_report_export_produces_html_and_pdf(
    tmp_path: Path, config: Config
) -> None:
    original_path = _write_dxf(
        _build_bracket_drawing(author="Aluno A"), tmp_path / "original.dxf"
    )
    copy_path = _write_dxf(
        _build_bracket_drawing(angle_degrees=20.0, scale=1.2, author="Aluno A"),
        tmp_path / "copy.dxf",
    )

    original_document = _read(original_path)
    copy_document = _read(copy_path)
    result = _run_pipeline(config, original_document, copy_document)

    report_generator = ReportGenerator(config)
    report: PairReport = report_generator.build_report(
        original_document, copy_document, result, name_a="original", name_b="copy"
    )

    output_dir = tmp_path / "reports"
    html_path, pdf_path = report_generator.export_report(
        report, output_dir, base_name="original_vs_copy"
    )

    assert html_path.exists()
    assert pdf_path.exists()
    assert html_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0

    html_content = html_path.read_text(encoding="utf-8")
    assert "original" in html_content
    assert "copy" in html_content
    assert f"{result.total_score:.1f}" in html_content


def test_sequence_analyzer_reflects_entity_order_from_disk(
    tmp_path: Path, config: Config
) -> None:
    path = _write_dxf(_build_bracket_drawing(), tmp_path / "a.dxf")
    document = _read(path)

    sequence_analyzer = SequenceAnalyzer(config)
    sequence = sequence_analyzer.build_sequence(document)

    assert sequence == "LLLLCAPITMD"
