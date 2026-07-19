"""Script para guiar a instalacao do ODA File Converter.

Uso:
    python scripts/install_oda.py           # modo interativo
    python scripts/install_oda.py --check   # apenas verifica se esta instalado
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

CONFIG_DIR: Path = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    if sys.platform != "win32"
    else Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
) / "plagio_dwg"
CONFIG_FILE = CONFIG_DIR / "oda_path"


def _get_common_paths() -> list[Path]:
    system = sys.platform
    paths: list[Path] = []
    if system == "linux":
        paths = [
            Path("/usr/bin/ODAFileConverter"),
            Path("/usr/local/bin/ODAFileConverter"),
            Path("/opt/ODAFileConverter/ODAFileConverter"),
        ]
    elif system == "darwin":
        paths = [
            Path("/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"),
            Path.home() / "Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
        ]
    elif system == "win32":
        prog = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
        prog86 = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
        paths = [
            prog / "ODA" / "ODAFileConverter" / "ODAFileConverter.exe",
            prog86 / "ODA" / "ODAFileConverter" / "ODAFileConverter.exe",
        ]
    return paths


def find_oda() -> str | None:
    which = shutil.which("ODAFileConverter")
    if which:
        return which
    for path in _get_common_paths():
        if path.exists():
            return str(path)
    return None


def check() -> bool:
    path = find_oda()
    if path:
        print(f"ODA File Converter encontrado em: {path}")
        return True
    print("ODA File Converter nao encontrado.")
    return False


def _save_path(path: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(path.strip() + "\n")
    print(f"Caminho salvo em: {CONFIG_FILE}")


def _test_converter(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode in (0, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def interactive() -> None:
    print("=" * 60)
    print("  Instalacao do ODA File Converter")
    print("=" * 60)

    found = find_oda()
    if found:
        print(f"\nODA File Converter ja esta instalado em: {found}")
        _save_path(found)
        return

    print("\nO ODA File Converter e necessario para converter arquivos .dwg para .dxf.")
    print("Ele e gratuito e fornecido pela Open Design Alliance.\n")

    answer = input("Deseja abrir o site de download no navegador? (S/n): ").strip().lower()
    if answer != "n":
        url = "https://www.opendesign.com/guestfiles/oda_file_converter"
        print(f"Abrindo: {url}")
        webbrowser.open(url)

    print("\nApos baixar e instalar, informe o caminho do executavel.")
    print("(Deixe em branco para procurar em locais comuns)\n")

    typed = input("Caminho do ODAFileConverter: ").strip()
    if typed:
        path = typed
    else:
        guessed = find_oda()
        if guessed:
            path = guessed
            print(f"Encontrado em: {path}")
        else:
            path = "ODAFileConverter"
            print("Mantendo o padrao 'ODAFileConverter' (assumindo que estara no PATH).")

    if path != "ODAFileConverter" and not _test_converter(path):
        print("Aviso: o caminho informado nao parece ser um executavel valido.")
        proceed = input("Deseja salvar mesmo assim? (s/N): ").strip().lower()
        if proceed != "s":
            print("Nada foi salvo.")
            return

    _save_path(path)
    print("\nInstalacao concluida. O DwgConverter usara este caminho automaticamente.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guia de instalacao do ODA File Converter")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Apenas verifica se o ODA File Converter esta instalado",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Caminho do ODA File Converter para salvar no config",
    )
    args = parser.parse_args()

    if args.check:
        return 0 if check() else 1

    if args.path:
        resolved = shutil.which(args.path) or args.path
        _save_path(resolved)
        print(f"Configurado: {resolved}")
        return 0

    interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
