from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import Config
from detector.features import FeatureExtractor
from detector.graph import GraphBuilder
from detector.normalizer import Normalizer
from detector.reader import CadDocument, DxfReader
from detector.report import PairReport, ReportGenerator
from detector.sequence import SequenceAnalyzer
from detector.similarity import SimilarityEngine, SimilarityResult


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
        paths = sorted(self._folder.glob("*.dxf"))
        if len(paths) < 2:
            raise ValueError(
                "Selecione uma pasta com ao menos 2 arquivos .dxf para comparar."
            )

        reader = DxfReader()
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

        pairs = list(combinations(paths, 2))
        results: List[PairResult] = []
        for index, (path_a, path_b) in enumerate(pairs, start=1):
            self.progress.emit(
                index, len(pairs), f"Comparando {path_a.name} x {path_b.name}"
            )
            sequence_result = sequence_analyzer.compare(
                sequences[path_a], sequences[path_b]
            )
            similarity_result = similarity_engine.compute_score(
                features[path_a],
                features[path_b],
                sequence_result,
                graphs[path_a],
                graphs[path_b],
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
        self.resize(1000, 750)

        layout = QVBoxLayout(self)

        score_label = QLabel(
            f"<b>Índice de suspeita: {report.similarity_result.total_score:.1f} / 100</b>"
        )
        score_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(score_label)

        justification_label = QLabel(report.justification)
        justification_label.setWordWrap(True)
        layout.addWidget(justification_label)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap()
        pixmap.loadFromData(report.comparison_image_png)
        image_label.setPixmap(
            pixmap.scaledToWidth(950, Qt.TransformationMode.SmoothTransformation)
        )
        layout.addWidget(image_label)

        metrics_table = QTableWidget()
        self._populate_metrics_table(metrics_table)
        layout.addWidget(metrics_table)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        export_button = QPushButton("Exportar PDF")
        export_button.clicked.connect(self._on_export_pdf)
        button_row.addWidget(export_button)
        layout.addLayout(button_row)

    def _populate_metrics_table(self, table: QTableWidget) -> None:
        dataframe = self._report.metrics_table
        table.setColumnCount(len(dataframe.columns))
        table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])
        table.setRowCount(len(dataframe))
        for row_index, row in enumerate(dataframe.itertuples(index=False)):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._config = config if config is not None else Config()
        self._report_generator = ReportGenerator(self._config)
        self._selected_folder: Optional[Path] = None
        self._results: List[PairResult] = []
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
        self._process_button = QPushButton("Processar")
        self._process_button.setEnabled(False)
        self._process_button.clicked.connect(self._on_process)
        selection_row.addWidget(self._select_folder_button)
        selection_row.addWidget(self._folder_label, 1)
        selection_row.addWidget(self._process_button)
        layout.addLayout(selection_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._ranking_table = QTableWidget()
        self._ranking_table.setColumnCount(4)
        self._ranking_table.setHorizontalHeaderLabels(
            ["Aluno A", "Aluno B", "Score de Suspeita", "Justificativa"]
        )
        self._ranking_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._ranking_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._ranking_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ranking_table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self._ranking_table)

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta com arquivos DXF"
        )
        if not folder:
            return
        self._selected_folder = Path(folder)
        self._folder_label.setText(str(self._selected_folder))
        self._process_button.setEnabled(True)

    def _on_process(self) -> None:
        if self._selected_folder is None:
            return
        self._process_button.setEnabled(False)
        self._select_folder_button.setEnabled(False)
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
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"{len(results)} par(es) comparado(s).")
        self._process_button.setEnabled(True)
        self._select_folder_button.setEnabled(True)
        self._populate_ranking_table()

    def _on_failed(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText("Falha no processamento.")
        self._process_button.setEnabled(True)
        self._select_folder_button.setEnabled(True)
        QMessageBox.critical(self, "Erro ao processar", message)

    def _populate_ranking_table(self) -> None:
        self._ranking_table.setRowCount(len(self._results))
        for row_index, result in enumerate(self._results):
            justification = self._report_generator.build_justification(
                result.similarity_result
            )
            self._ranking_table.setItem(
                row_index, 0, QTableWidgetItem(result.path_a.stem)
            )
            self._ranking_table.setItem(
                row_index, 1, QTableWidgetItem(result.path_b.stem)
            )
            self._ranking_table.setItem(
                row_index,
                2,
                QTableWidgetItem(f"{result.similarity_result.total_score:.1f}"),
            )
            self._ranking_table.setItem(
                row_index, 3, QTableWidgetItem(justification)
            )

    def _on_row_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._results):
            return
        result = self._results[row]
        key = (result.path_a, result.path_b)
        report = self._report_cache.get(key)
        if report is None:
            report = self._report_generator.build_report(
                result.document_a,
                result.document_b,
                result.similarity_result,
                name_a=result.path_a.stem,
                name_b=result.path_b.stem,
            )
            self._report_cache[key] = report
        dialog = ComparisonDialog(report, self._report_generator, self)
        dialog.exec()
