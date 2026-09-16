from __future__ import annotations

import base64
import io
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import pandas as pd
from matplotlib import image as mpimg
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from config import Config
from detector.corpus import CorpusContext, percentile_rank, z_score
from detector.reader import (
    ArcEntity,
    CadDocument,
    CadEntity,
    CircleEntity,
    InsertEntity,
    LineEntity,
    MTextEntity,
    PolylineEntity,
    TextEntity,
)
from detector.similarity import SimilarityResult


_COMPONENT_LABELS: Dict[str, str] = {
    "geometry": "Geometria",
    "sequence": "Sequência",
    "graph": "Grafo",
    "dimensions": "Cotas",
    "text": "Texto",
    "blocks": "Blocos",
    "styles": "Estilos",
    "decimal_precision": "Precisão Decimal",
    "pieces": "Peças (casamento geométrico)",
}

_CORPUS_OUTLIER_Z_SCORE = 2.0
_CORPUS_TYPICAL_Z_SCORE = 1.0
_CORPUS_LOW_Z_SCORE = -1.0
_MIN_CORPUS_SAMPLE_SIZE = 2

_METRICS_TABLE_COLUMNS: Tuple[str, ...] = (
    "componente",
    "peso",
    "contribuicao",
    "score",
)

SEVERITY_COPY = "copy"
SEVERITY_VERY_HIGH = "very_high"
SEVERITY_HIGH = "high"
SEVERITY_MODERATE = "moderate"
SEVERITY_LOW = "low"

_SEVERITY_LABELS: Dict[str, str] = {
    SEVERITY_COPY: "Cópia",
    SEVERITY_VERY_HIGH: "Muito alta",
    SEVERITY_HIGH: "Alta",
    SEVERITY_MODERATE: "Moderada",
    SEVERITY_LOW: "Baixa",
}

_ABSOLUTE_HEADLINE_TEXT: Dict[str, str] = {
    SEVERITY_VERY_HIGH: "Suspeita muito alta de cópia. Recomenda-se revisão manual detalhada.",
    SEVERITY_HIGH: "Suspeita alta de cópia. Recomenda-se revisão manual.",
    SEVERITY_MODERATE: "Suspeita moderada. Pode refletir a semelhança natural do exercício.",
    SEVERITY_LOW: "Baixa suspeita. Compatível com desenhos elaborados de forma independente.",
}

_CORPUS_HEADLINE_TEXT: Dict[str, str] = {
    SEVERITY_VERY_HIGH: (
        "Suspeita muito alta de cópia — esta dupla é um forte destaque (outlier) em "
        "relação ao restante da turma. Recomenda-se revisão manual detalhada."
    ),
    SEVERITY_HIGH: (
        "Suspeita alta de cópia — a similaridade desta dupla está acima do padrão "
        "observado na turma. Recomenda-se revisão manual."
    ),
    SEVERITY_MODERATE: (
        "Suspeita moderada a baixa — a similaridade está dentro do padrão observado "
        "na turma, compatível com um gabarito, modelo ou exercício compartilhado por "
        "todos, não necessariamente cópia entre esta dupla."
    ),
    SEVERITY_LOW: "Baixa suspeita — esta dupla é menos similar que a média da turma.",
}


def suspicion_severity(total_score: float, corpus_z_score: Optional[float] = None) -> str:
    if total_score >= 100.0:
        return SEVERITY_COPY
    if corpus_z_score is not None:
        if corpus_z_score >= _CORPUS_OUTLIER_Z_SCORE:
            return SEVERITY_VERY_HIGH
        if corpus_z_score >= _CORPUS_TYPICAL_Z_SCORE:
            return SEVERITY_HIGH
        if corpus_z_score >= _CORPUS_LOW_Z_SCORE:
            return SEVERITY_MODERATE
        return SEVERITY_LOW
    if total_score >= 80.0:
        return SEVERITY_VERY_HIGH
    if total_score >= 60.0:
        return SEVERITY_HIGH
    if total_score >= 40.0:
        return SEVERITY_MODERATE
    return SEVERITY_LOW


def suspicion_label(total_score: float, corpus_z_score: Optional[float] = None) -> str:
    return _SEVERITY_LABELS[suspicion_severity(total_score, corpus_z_score)]


@dataclass(frozen=True, eq=False)
class PairReport:
    document_a_name: str
    document_b_name: str
    similarity_result: SimilarityResult
    justification: str
    metrics_table: pd.DataFrame
    comparison_image_png: bytes
    corpus_sample_size: Optional[int] = None
    corpus_percentile: Optional[float] = None
    corpus_z_score: Optional[float] = None


def _draw_entity(ax: Axes, entity: CadEntity) -> None:
    if isinstance(entity, LineEntity):
        ax.plot(
            [entity.start[0], entity.end[0]],
            [entity.start[1], entity.end[1]],
            color="black",
            linewidth=0.8,
        )
    elif isinstance(entity, CircleEntity):
        ax.add_patch(
            patches.Circle(
                (entity.center[0], entity.center[1]),
                entity.radius,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
            )
        )
    elif isinstance(entity, ArcEntity):
        diameter = entity.radius * 2.0
        ax.add_patch(
            patches.Arc(
                (entity.center[0], entity.center[1]),
                diameter,
                diameter,
                angle=0.0,
                theta1=entity.start_angle,
                theta2=entity.end_angle,
                edgecolor="black",
                linewidth=0.8,
            )
        )
    elif isinstance(entity, PolylineEntity):
        xs = [point[0] for point in entity.points]
        ys = [point[1] for point in entity.points]
        if entity.closed and entity.points:
            xs.append(entity.points[0][0])
            ys.append(entity.points[0][1])
        ax.plot(xs, ys, color="black", linewidth=0.8)
    elif isinstance(entity, InsertEntity):
        ax.plot(
            entity.insert_point[0],
            entity.insert_point[1],
            marker="s",
            color="blue",
            markersize=3,
        )
    elif isinstance(entity, (TextEntity, MTextEntity)):
        ax.plot(
            entity.insert_point[0],
            entity.insert_point[1],
            marker=".",
            color="red",
            markersize=3,
        )


def _justification_headline(total_score: float, corpus_z_score: Optional[float] = None) -> str:
    severity = suspicion_severity(total_score, corpus_z_score)
    if severity == SEVERITY_COPY:
        return "cópia"
    if corpus_z_score is not None:
        return _CORPUS_HEADLINE_TEXT[severity]
    return _ABSOLUTE_HEADLINE_TEXT[severity]


def _score_color_class(score: float) -> str:
    if score < 70.0:
        return "score-green"
    if score < 80.0:
        return "score-yellow"
    if score < 95.0:
        return "score-orange"
    return "score-red"


def _score_facecolor(score: float) -> str:
    if score < 70.0:
        return "#d4edda"
    if score < 80.0:
        return "#fff3cd"
    if score < 95.0:
        return "#ffe0b2"
    return "#f8d7da"


def _format_metric(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.2f}"


_PDF_LINE_HEIGHT = 0.026
_PDF_WRAP_WIDTH_BY_FONTSIZE: Dict[int, int] = {
    16: 46,
    12: 62,
    11: 68,
    10: 76,
    9: 86,
}


def _wrap_for_fontsize(text: str, fontsize: int) -> List[str]:
    width = _PDF_WRAP_WIDTH_BY_FONTSIZE.get(fontsize, 80)
    return textwrap.wrap(text, width=width) or [""]


class ReportGenerator:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config if config is not None else Config()

    def build_metrics_table(self, similarity_result: SimilarityResult) -> pd.DataFrame:
        rows = [
            {
                "componente": _COMPONENT_LABELS.get(component.name, component.name),
                "peso": component.weight,
                "score": (
                    round(component.score * 100.0, 2)
                    if component.applicable
                    else float("nan")
                ),
                "contribuicao": (
                    round(component.score * component.weight * 100.0, 2)
                    if component.applicable
                    else float("nan")
                ),
            }
            for component in similarity_result.component_scores
        ]
        return pd.DataFrame(rows, columns=list(_METRICS_TABLE_COLUMNS))

    def build_justification(
        self,
        similarity_result: SimilarityResult,
        corpus_context: Optional[CorpusContext] = None,
    ) -> str:
        has_corpus = (
            corpus_context is not None
            and len(corpus_context.scores) >= _MIN_CORPUS_SAMPLE_SIZE
        )
        corpus_z = (
            z_score(similarity_result.total_score, corpus_context) if has_corpus else None
        )
        headline = _justification_headline(similarity_result.total_score, corpus_z)
        details = "; ".join(
            component.justification for component in similarity_result.component_scores
        )
        text = (
            f"Índice de suspeita: {round(similarity_result.total_score, 1)}/100. "
            f"{headline} Detalhamento por componente: {details}."
        )
        corpus_text = self._corpus_justification(similarity_result, corpus_context)
        if corpus_text:
            text += f" {corpus_text}"
        return text

    def _corpus_justification(
        self,
        similarity_result: SimilarityResult,
        corpus_context: Optional[CorpusContext],
    ) -> str:
        if corpus_context is None or len(corpus_context.scores) < _MIN_CORPUS_SAMPLE_SIZE:
            return ""
        sample_size = len(corpus_context.scores)
        percentile = percentile_rank(similarity_result.total_score, corpus_context)
        z = z_score(similarity_result.total_score, corpus_context)
        return (
            f"Em relação aos {sample_size} pares comparados neste lote, esta dupla está "
            f"no percentil {percentile:.0f} (Z = {z:.2f})."
        )

    def build_comparison_figure(
        self,
        document_a: CadDocument,
        document_b: CadDocument,
        name_a: str,
        name_b: str,
    ) -> Figure:
        figure = Figure(figsize=(10.0, 5.0), dpi=100)
        ax_a = figure.add_subplot(1, 2, 1)
        ax_b = figure.add_subplot(1, 2, 2)
        self._render_document(ax_a, document_a, name_a)
        self._render_document(ax_b, document_b, name_b)
        figure.tight_layout()
        return figure

    def render_comparison_image(
        self,
        document_a: CadDocument,
        document_b: CadDocument,
        name_a: str,
        name_b: str,
    ) -> bytes:
        figure = self.build_comparison_figure(document_a, document_b, name_a, name_b)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        return buffer.getvalue()

    def build_report(
        self,
        document_a: CadDocument,
        document_b: CadDocument,
        similarity_result: SimilarityResult,
        name_a: Optional[str] = None,
        name_b: Optional[str] = None,
        corpus_context: Optional[CorpusContext] = None,
    ) -> PairReport:
        resolved_name_a = name_a if name_a is not None else self._document_name(document_a, "Documento A")
        resolved_name_b = name_b if name_b is not None else self._document_name(document_b, "Documento B")
        has_corpus = (
            corpus_context is not None
            and len(corpus_context.scores) >= _MIN_CORPUS_SAMPLE_SIZE
        )
        return PairReport(
            document_a_name=resolved_name_a,
            document_b_name=resolved_name_b,
            similarity_result=similarity_result,
            justification=self.build_justification(similarity_result, corpus_context),
            metrics_table=self.build_metrics_table(similarity_result),
            comparison_image_png=self.render_comparison_image(
                document_a, document_b, resolved_name_a, resolved_name_b
            ),
            corpus_sample_size=len(corpus_context.scores) if has_corpus else None,
            corpus_percentile=(
                percentile_rank(similarity_result.total_score, corpus_context)
                if has_corpus
                else None
            ),
            corpus_z_score=(
                z_score(similarity_result.total_score, corpus_context)
                if has_corpus
                else None
            ),
        )

    def generate_html(self, report: PairReport) -> str:
        image_base64 = base64.b64encode(report.comparison_image_png).decode("ascii")
        table_rows = self._html_table_rows(report.metrics_table)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="pt-br">\n'
            "<head>\n"
            '<meta charset="utf-8" />\n'
            f"<title>Relatório de Similaridade - {report.document_a_name} x {report.document_b_name}</title>\n"
            "<style>\n"
            "body { font-family: Arial, sans-serif; margin: 24px; color: #1a1a1a; }\n"
            "h1 { font-size: 20px; }\n"
            ".score { font-size: 32px; font-weight: bold; }\n"
            "table { border-collapse: collapse; width: 100%; margin-top: 12px; }\n"
            "th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px; }\n"
            "th { background-color: #f0f0f0; }\n"
            ".score-green { background-color: #d4edda; color: #155724; font-weight: bold; }\n"
            ".score-yellow { background-color: #fff3cd; color: #856404; font-weight: bold; }\n"
            ".score-orange { background-color: #ffe0b2; color: #e65100; font-weight: bold; }\n"
            ".score-red { background-color: #f8d7da; color: #721c24; font-weight: bold; }\n"
            ".score-na { background-color: #e9ecef; color: #495057; font-weight: bold; }\n"
            "img { max-width: 100%; margin-top: 16px; border: 1px solid #ccc; }\n"
            ".justificativa { margin-top: 16px; line-height: 1.5; }\n"
            "</style>\n"
            "</head>\n"
            "<body>\n"
            f"<h1>Relatório de Similaridade: {report.document_a_name} × {report.document_b_name}</h1>\n"
            f'<div class="score">Índice de suspeita: {report.similarity_result.total_score:.1f} / 100</div>\n'
            f'<div class="justificativa">{report.justification}</div>\n'
            f"{table_rows}\n"
            f'<img src="data:image/png;base64,{image_base64}" alt="Comparação lado a lado" />\n'
            "</body>\n"
            "</html>\n"
        )

    def _html_table_rows(self, table: pd.DataFrame) -> str:
        rows = ["<table>"]
        rows.append("<tr><th>Componente</th><th>Peso</th><th>Contribuição (%)</th><th>Score (%)</th></tr>")
        for _, row in table.iterrows():
            score = row["score"]
            contrib = row["contribuicao"]
            score_class = "score-na" if math.isnan(score) else _score_color_class(score)
            contrib_class = "score-na" if math.isnan(contrib) else _score_color_class(contrib)
            rows.append(
                f"<tr>"
                f"<td>{row['componente']}</td>"
                f"<td>{row['peso']:.2f}</td>"
                f'<td class="{contrib_class}">{_format_metric(contrib)}</td>'
                f'<td class="{score_class}">{_format_metric(score)}</td>'
                f"</tr>"
            )
        rows.append("</table>")
        return "\n".join(rows)

    def export_html(self, report: PairReport, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_html(report), encoding="utf-8")
        return output_path

    def export_pdf(self, report: PairReport, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output_path) as pdf:
            pdf.savefig(self._build_summary_figure(report))
            pdf.savefig(self._build_table_figure(report))
            pdf.savefig(self._build_image_figure(report))
        return output_path

    def export_report(
        self, report: PairReport, output_dir: Optional[Path], base_name: str
    ) -> Tuple[Path, Path]:
        resolved_output_dir = output_dir if output_dir is not None else self._config.output_dir
        html_path = self.export_html(report, resolved_output_dir / f"{base_name}.html")
        pdf_path = self.export_pdf(report, resolved_output_dir / f"{base_name}.pdf")
        return html_path, pdf_path

    def _document_name(self, document: CadDocument, fallback: str) -> str:
        if document.source_path is not None:
            return document.source_path.stem
        return fallback

    def _render_document(self, ax: Axes, document: CadDocument, title: str) -> None:
        for entity in document.entities:
            _draw_entity(ax, entity)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    def _draw_wrapped_text(
        self,
        ax: Axes,
        text: str,
        y: float,
        fontsize: int = 9,
        indent: float = 0.0,
        italic: bool = False,
        bold: bool = False,
    ) -> float:
        for line in _wrap_for_fontsize(text, fontsize):
            ax.text(
                indent, y, line,
                fontsize=fontsize,
                fontstyle="italic" if italic else "normal",
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes, va="top",
            )
            y -= _PDF_LINE_HEIGHT
        return y

    def _build_summary_figure(self, report: PairReport) -> Figure:
        figure = Figure(figsize=(8.27, 11.69), dpi=100)
        ax = figure.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(
            f"Relatório de Similaridade\n{report.document_a_name} × {report.document_b_name}",
            fontsize=13, fontweight="bold",
        )
        y = 0.92
        ax.text(
            0.0, y,
            f"Índice de suspeita: {report.similarity_result.total_score:.1f} / 100",
            fontsize=16, fontweight="bold", transform=ax.transAxes, va="top",
        )
        y -= 0.05

        suspicion = _justification_headline(
            report.similarity_result.total_score, report.corpus_z_score
        )
        y = self._draw_wrapped_text(ax, suspicion, y, fontsize=11)
        y -= 0.015

        if report.corpus_sample_size is not None:
            corpus_line = (
                f"Lote: percentil {report.corpus_percentile:.0f} entre "
                f"{report.corpus_sample_size} pares (Z = {report.corpus_z_score:.2f})."
            )
            y = self._draw_wrapped_text(ax, corpus_line, y, fontsize=9, italic=True)
        y -= 0.03

        y = self._draw_wrapped_text(ax, "Detalhamento por componente:", y, fontsize=11, bold=True)
        y -= 0.01
        for component in report.similarity_result.component_scores:
            y = self._draw_wrapped_text(
                ax, f"•  {component.justification}", y, fontsize=9, indent=0.015
            )
            y -= 0.008
        return figure

    def _build_table_figure(self, report: PairReport) -> Figure:
        figure = Figure(figsize=(8.27, 11.69), dpi=100)
        ax = figure.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(
            f"Detalhamento por Componente\n{report.document_a_name} × {report.document_b_name}",
            fontsize=13, fontweight="bold", pad=20,
        )
        table_data = [
            [
                row["componente"],
                f"{row['peso']:.2f}",
                _format_metric(row["contribuicao"]),
                _format_metric(row["score"]),
            ]
            for _, row in report.metrics_table.iterrows()
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["Componente", "Peso", "Contribuição (%)", "Score (%)"],
            loc="upper center",
            cellLoc="left",
            colWidths=[0.44, 0.10, 0.23, 0.23],
            bbox=[0.0, 0.15, 1.0, 0.75],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.4)
        self._color_table_cells(table)
        return figure

    def _color_table_cells(self, table) -> None:
        for row_idx in range(1, len(table.get_celld()) // 4):
            score_cell = table[row_idx, 3]
            score_text = score_cell.get_text().get_text()
            if score_text == "N/A":
                score_cell.set_facecolor("#e9ecef")
            else:
                score_cell.set_facecolor(_score_facecolor(float(score_text)))

    def _build_image_figure(self, report: PairReport) -> Figure:
        image_array = mpimg.imread(io.BytesIO(report.comparison_image_png), format="png")
        figure = Figure(figsize=(11.69, 8.27), dpi=100)
        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(image_array)
        ax.axis("off")
        return figure
