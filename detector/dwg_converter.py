from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


CONFIG_DIR: Path = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    if sys.platform != "win32"
    else Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
) / "plagio_dwg"
CONFIG_FILE = CONFIG_DIR / "oda_path"


def _resolve_converter_path(explicit: str) -> str:
    if explicit != "ODAFileConverter":
        return explicit
    if CONFIG_FILE.exists():
        stored = CONFIG_FILE.read_text().strip()
        if stored:
            return stored
    return explicit


_INSTALL_HINT = """
O ODA File Converter e necessario para ler arquivos .dwg.
Instale-o e configure com:
    python scripts/install_oda.py
ou baixe diretamente de:
    https://www.opendesign.com/guestfiles/oda_file_converter
"""

_XVFB_HINT = (
    "\nO xvfb-run e necessario para rodar o ODA File Converter sem abrir janela.\n"
    "Instale com:\n    sudo apt install xvfb\n"
) if sys.platform == "linux" else ""


class DwgConversionError(Exception):
    pass


class DwgConverter:
    def __init__(
        self,
        converter_path: str = "ODAFileConverter",
        output_version: str = "ACAD2018",
        output_format: str = "DXF",
        temp_dir: Optional[Path] = None,
    ) -> None:
        self._converter_path = _resolve_converter_path(converter_path)
        self._output_version = output_version
        self._output_format = output_format
        self._temp_dir = temp_dir

    def convert(self, dwg_path: Path) -> Path:
        dwg_path = dwg_path.resolve()
        if not dwg_path.exists():
            raise DwgConversionError(f"Arquivo nao encontrado: {dwg_path}")

        if not self.is_available():
            raise DwgConversionError(
                f"Conversor nao encontrado: {self._converter_path}" + _INSTALL_HINT
            )

        temp_root = Path(tempfile.mkdtemp(dir=self._temp_dir))
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        dest = input_dir / dwg_path.name
        shutil.copy2(str(dwg_path), str(dest))

        dxf_name = dwg_path.stem + ".dxf"
        dxf_path = output_dir / dxf_name

        try:
            self._run_converter(input_dir, output_dir)
        except Exception as exc:
            shutil.rmtree(str(temp_root), ignore_errors=True)
            raise DwgConversionError(
                f"Falha ao converter {dwg_path.name}: {exc}"
            ) from exc

        if not dxf_path.exists():
            candidates = list(output_dir.glob("*.dxf"))
            if not candidates:
                shutil.rmtree(str(temp_root), ignore_errors=True)
                raise DwgConversionError(
                    f"Arquivo DXF nao gerado para {dwg_path.name}"
                )
            dxf_path = candidates[0]

        return dxf_path

    def _run_converter(self, input_dir: Path, output_dir: Path) -> None:
        xvfb = shutil.which("xvfb-run")
        cmd = [xvfb, "-a", self._converter_path] if xvfb else [self._converter_path]
        cmd += [
            str(input_dir),
            str(output_dir),
            self._output_version,
            self._output_format,
            "0",
            "0",
            "*.dwg",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            msg = result.stderr.strip() or result.stdout.strip()
            if not xvfb:
                msg += _XVFB_HINT
            raise DwgConversionError(
                f"ODAFileConverter retornou codigo {result.returncode}: {msg}"
            )

    def is_available(self) -> bool:
        return shutil.which(self._converter_path) is not None
