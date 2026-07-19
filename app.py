from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="Detector de Possível Plágio em Desenhos CAD")
    parser.add_argument(
        "-f", "--folder",
        type=Path,
        default=None,
        help="Pasta com arquivos DXF/DWG para processar automaticamente ao iniciar",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(initial_folder=args.folder)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
