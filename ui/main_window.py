from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import Config
from detector.corpus import CorpusContext, build_corpus_context, z_score
from detector.features import FeatureExtractor
from detector.graph import GraphBuilder
from detector.normalizer import Normalizer
from detector.reader import CadDocument, CadReader
from detector.report import PairReport, ReportGenerator, suspicion_label, suspicion_severity
from detector.sequence import SequenceAnalyzer
from detector.similarity import SimilarityEngine, SimilarityResult


def _vibrant_score_color(score: float) -> str:
    if score < 70.0:
        return "#27ae60"
    if score < 80.0:
        return "#f1c40f"
    if score < 95.0:
        return "#e67e22"
    return "#e74c3c"


_SEVERITY_COLORS: Dict[str, str] = {
    "copy": "#8e1b0f",
    "very_high": "#e74c3c",
    "high": "#e67e22",
    "moderate": "#f1c40f",
    "low": "#27ae60",
}


def _severity_color(severity: str) -> str:
    return _SEVERITY_COLORS.get(severity, "#95a5a6")


@dataclass(frozen=True)
class PairResult:
    path_a: Path
    path_b: Path
    document_a: CadDocument
    document_b: CadDocument
    similarity_result: SimilarityResult


class ProcessingWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self, folder: Path, config: Config, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._folder = folder
        self._config = config

    def run(self) -> None:
        try:
            results = self._process()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(results)

    def _process(self) -> List[PairResult]:
        paths = sorted(
            set(self._folder.glob("*.dxf")) | set(self._folder.glob("*.dwg"))
        )
        if len(paths) < 2:
            raise ValueError(
                "Selecione uma pasta com ao menos 2 arquivos .dxf ou .dwg para comparar."
            )

        reader = CadReader()
        normalizer = Normalizer()
        feature_extractor = FeatureExtractor(self._config)
        graph_builder = GraphBuilder(self._config)
        sequence_analyzer = SequenceAnalyzer(self._config)
        similarity_engine = SimilarityEngine(self._config)

        documents: Dict[Path, CadDocument] = {}
        for index, path in enumerate(paths, start=1):
            self.progress.emit(index, len(paths), f"Lendo {path.name}")
            documents[path] = normalizer.normalize(reader.read(path))

        features = {
            path: feature_extractor.extract(document)
            for path, document in documents.items()
        }
        graphs = {
            path: graph_builder.compute_metrics(graph_builder.build(document))
            for path, document in documents.items()
        }
        sequences = {
            path: sequence_analyzer.build_sequence(document)
            for path, document in documents.items()
        }
        text_sequences = {
            path: sequence_analyzer.build_text_sequence(document)
            for path, document in documents.items()
        }

        pairs = list(combinations(paths, 2))
        results: List[PairResult] = []
        for index, (path_a, path_b) in enumerate(pairs, start=1):
            self.progress.emit(
                index, len(pairs), f"Comparando {path_a.name} x {path_b.name}"
            )
            sequence_result = sequence_analyzer.compare(
                sequences[path_a], sequences[path_b]
            )
            text_sequence_result = sequence_analyzer.compare_text_sequence(
                text_sequences[path_a], text_sequences[path_b]
            )
            similarity_result = similarity_engine.compute_score(
                features[path_a],
                features[path_b],
                sequence_result,
                graphs[path_a],
                graphs[path_b],
                text_sequence_result,
                documents[path_a],
                documents[path_b],
            )
            results.append(
                PairResult(
                    path_a=path_a,
                    path_b=path_b,
                    document_a=documents[path_a],
                    document_b=documents[path_b],
                    similarity_result=similarity_result,
                )
            )
        results.sort(key=lambda result: result.similarity_result.total_score, reverse=True)
        return results


class ComparisonDialog(QDialog):
    def __init__(
        self,
        report: PairReport,
        report_generator: ReportGenerator,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._report = report
        self._report_generator = report_generator

        self.setWindowTitle(
            f"Comparação: {report.document_a_name} × {report.document_b_name}"
        )
        self.resize(1050, 820)

        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        outer_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)

        layout.addLayout(self._build_header_row(report))

        justification_label = QLabel(report.justification)
        justification_label.setWordWrap(True)
        layout.addWidget(justification_label)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap()
        pixmap.loadFromData(report.comparison_image_png)
        image_label.setPixmap(
            pixmap.scaledToWidth(980, Qt.TransformationMode.SmoothTransformation)
        )
        layout.addWidget(image_label)

        table_title = QLabel("<b>Detalhamento por componente</b>")
        table_title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(table_title)

        metrics_table = QTableWidget()
        self._populate_metrics_table(metrics_table)
        metrics_table.setMinimumHeight(340)
        layout.addWidget(metrics_table)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        export_button = QPushButton("Exportar PDF")
        export_button.clicked.connect(self._on_export_pdf)
        button_row.addWidget(export_button)
        layout.addLayout(button_row)

    def _build_header_row(self, report: PairReport) -> QHBoxLayout:
        header_row = QHBoxLayout()

        score_label = QLabel(
            f"<span style='font-size:22px; font-weight:bold;'>"
            f"{report.similarity_result.total_score:.1f} / 100</span>"
        )
        score_label.setTextFormat(Qt.TextFormat.RichText)
        header_row.addWidget(score_label)

        severity = suspicion_severity(
            report.similarity_result.total_score, report.corpus_z_score
        )
        verdict_label = QLabel(suspicion_label(
            report.similarity_result.total_score, report.corpus_z_score
        ))
        verdict_label.setStyleSheet(
            f"background-color: {_severity_color(severity)}; color: #ffffff; "
            "padding: 3px 10px; border-radius: 4px; font-weight: bold;"
        )
        header_row.addWidget(verdict_label)

        if report.corpus_sample_size is not None:
            corpus_label = QLabel(
                f"Percentil {report.corpus_percentile:.0f} de "
                f"{report.corpus_sample_size} pares do lote (Z = {report.corpus_z_score:.2f})"
            )
            corpus_label.setStyleSheet("color: #555555; font-style: italic;")
            header_row.addWidget(corpus_label)

        header_row.addStretch(1)
        return header_row

    def _populate_metrics_table(self, table: QTableWidget) -> None:
        dataframe = self._report.metrics_table
        table.setColumnCount(len(dataframe.columns))
        table.setHorizontalHeaderLabels(
            ["Componente", "Peso", "Contribuição (%)", "Score (%)"]
        )
        table.setRowCount(len(dataframe))
        for row_index, row in enumerate(dataframe.itertuples(index=False)):
            for column_index, value in enumerate(row):
                is_na = isinstance(value, float) and math.isnan(value)
                if is_na:
                    text = "N/A"
                elif isinstance(value, float):
                    text = f"{value:.2f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if column_index == 3:
                    color = "#e9ecef" if is_na else _vibrant_score_color(float(value))
                    item.setBackground(QColor(color))
                    item.setForeground(QColor("#000000"))
                table.setItem(row_index, column_index, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column_index in (1, 2, 3):
            header.setSectionResizeMode(column_index, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def _on_export_pdf(self) -> None:
        default_name = f"{self._report.document_a_name}_vs_{self._report.document_b_name}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Relatório PDF", default_name, "PDF (*.pdf)"
        )
        if not file_path:
            return
        try:
            self._report_generator.export_pdf(self._report, Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Erro ao exportar", str(exc))
            return
        QMessageBox.information(self, "Exportado", f"Relatório exportado para:\n{file_path}")


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: Optional[Config] = None,
        initial_folder: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config if config is not None else Config()
        self._report_generator = ReportGenerator(self._config)
        self._selected_folder: Optional[Path] = None
        self._results: List[PairResult] = []
        self._corpus_context: Optional[CorpusContext] = None
        self._report_cache: Dict[Tuple[Path, Path], PairReport] = {}
        self._worker: Optional[ProcessingWorker] = None

        self.setWindowTitle("Detector de Possível Plágio em Desenhos CAD")
        self.resize(1000, 640)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        selection_row = QHBoxLayout()
        self._select_folder_button = QPushButton("Selecionar Pasta")
        self._select_folder_button.clicked.connect(self._on_select_folder)
        self._folder_label = QLabel("Nenhuma pasta selecionada")
        selection_row.addWidget(self._select_folder_button)
        selection_row.addWidget(self._folder_label, 1)
        layout.addLayout(selection_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._batch_summary_label = QLabel("")
        self._batch_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self._batch_summary_label.setWordWrap(True)
        self._batch_summary_label.setStyleSheet(
            "background-color: #f4f4f4; padding: 6px 10px; border-radius: 4px;"
        )
        self._batch_summary_label.setVisible(False)
        layout.addWidget(self._batch_summary_label)

        self._ranking_table = QTableWidget()
        self._ranking_table.setColumnCount(4)
        self._ranking_table.setHorizontalHeaderLabels(
            ["Aluno A", "Aluno B", "Score de Suspeita", "Veredito (relativo à turma)"]
        )
        header = self._ranking_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._ranking_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._ranking_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ranking_table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self._ranking_table)

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self._export_all_button = QPushButton("Exportar Todos os Relatórios")
        self._export_all_button.setEnabled(False)
        self._export_all_button.clicked.connect(self._on_export_all)
        export_row.addWidget(self._export_all_button)
        layout.addLayout(export_row)

        if initial_folder is not None:
            self._selected_folder = initial_folder
            self._folder_label.setText(str(self._selected_folder))
            QTimer.singleShot(100, self._on_process)

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta com arquivos DXF/DWG"
        )
        if not folder:
            return
        self._selected_folder = Path(folder)
        self._folder_label.setText(str(self._selected_folder))
        QTimer.singleShot(50, self._on_process)

    def _on_process(self) -> None:
        if self._selected_folder is None:
            return
        self._select_folder_button.setEnabled(False)
        self._export_all_button.setEnabled(False)
        self._batch_summary_label.setVisible(False)
        self._ranking_table.setRowCount(0)
        self._report_cache.clear()
        self._results = []
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Processando...")

        self._worker = ProcessingWorker(self._selected_folder, self._config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)
        self._status_label.setText(message)

    def _on_finished(self, results: List[PairResult]) -> None:
        self._results = results
        self._corpus_context = build_corpus_context(
            [result.similarity_result.total_score for result in results]
        )
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"{len(results)} par(es) comparado(s).")
        self._select_folder_button.setEnabled(True)
        self._export_all_button.setEnabled(bool(results))
        self._update_batch_summary()
        self._populate_ranking_table()

    def _on_failed(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Falha no processamento.")
        self._select_folder_button.setEnabled(True)
        QMessageBox.critical(self, "Erro ao processar", message)

    def _update_batch_summary(self) -> None:
        context = self._corpus_context
        if context is None or len(context.scores) < 2:
            self._batch_summary_label.setVisible(False)
            return
        outliers = sum(1 for score in context.scores if z_score(score, context) >= 2.0)
        self._batch_summary_label.setText(
            f"<b>{len(context.scores)} pares comparados</b> neste lote — "
            f"média {context.mean:.1f}, desvio padrão {context.std:.1f}. "
            f"{outliers} par(es) com Z ≥ 2 (forte destaque/outlier em relação à turma)."
        )
        self._batch_summary_label.setVisible(True)

    def _populate_ranking_table(self) -> None:
        self._ranking_table.setRowCount(len(self._results))
        for row_index, result in enumerate(self._results):
            score = result.similarity_result.total_score
            corpus_z = (
                z_score(score, self._corpus_context)
                if self._corpus_context is not None and len(self._corpus_context.scores) >= 2
                else None
            )
            self._ranking_table.setItem(
                row_index, 0, QTableWidgetItem(result.path_a.stem)
            )
            self._ranking_table.setItem(
                row_index, 1, QTableWidgetItem(result.path_b.stem)
            )
            score_item = QTableWidgetItem(f"{score:.1f}")
            score_item.setBackground(QColor(_vibrant_score_color(score)))
            score_item.setForeground(QColor("#000000"))
            self._ranking_table.setItem(row_index, 2, score_item)

            severity = suspicion_severity(score, corpus_z)
            verdict_item = QTableWidgetItem(suspicion_label(score, corpus_z))
            verdict_item.setBackground(QColor(_severity_color(severity)))
            verdict_item.setForeground(QColor("#ffffff"))
            self._ranking_table.setItem(row_index, 3, verdict_item)

    def _get_or_build_report(self, result: PairResult) -> PairReport:
        key = (result.path_a, result.path_b)
        report = self._report_cache.get(key)
        if report is None:
            report = self._report_generator.build_report(
                result.document_a,
                result.document_b,
                result.similarity_result,
                name_a=result.path_a.stem,
                name_b=result.path_b.stem,
                corpus_context=self._corpus_context,
            )
            self._report_cache[key] = report
        return report

    def _on_row_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._results):
            return
        report = self._get_or_build_report(self._results[row])
        dialog = ComparisonDialog(report, self._report_generator, self)
        dialog.exec()

    def _on_export_all(self) -> None:
        if not self._results:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta para exportar todos os relatórios"
        )
        if not folder:
            return
        output_dir = Path(folder)
        errors: List[str] = []
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            for index, result in enumerate(self._results, start=1):
                self._status_label.setText(
                    f"Exportando {index}/{len(self._results)}: "
                    f"{result.path_a.stem} × {result.path_b.stem}"
                )
                QApplication.processEvents()
                report = self._get_or_build_report(result)
                base_name = f"{result.path_a.stem}_vs_{result.path_b.stem}"
                try:
                    self._report_generator.export_report(report, output_dir, base_name)
                except Exception as exc:
                    errors.append(f"{base_name}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._status_label.setText(f"{len(self._results)} par(es) comparado(s).")
        if errors:
            QMessageBox.warning(
                self, "Exportação concluída com erros",
                f"{len(self._results) - len(errors)} relatório(s) exportado(s).\n\n"
                "Falhas:\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(
                self, "Exportado",
                f"{len(self._results)} relatório(s) exportado(s) para:\n{output_dir}",
            )
