import argparse
import subprocess
import sys
from pathlib import Path


def _find_libexpat() -> Path | None:
    for prefix in (Path(sys.base_prefix), Path(sys.prefix)):
        lib_dir = prefix / "lib"
        so_path = lib_dir / "libexpat.so.1"
        if so_path.exists():
            return so_path.resolve()
        for so in sorted(lib_dir.rglob("libexpat.so.1*")):
            if so.name.startswith("libexpat.so.1."):
                return so.resolve()
    return None


def _build_spec(libexpat: Path | None, target: str) -> str:
    libexpat_str = f"r'{libexpat}'" if libexpat is not None else "None"
    libexpat_filter = ""
    if target == "linux" and libexpat is not None:
        libexpat_filter = f'''
if _libexpat_path:
    _new_binaries = []
    for _dest, _src, _typ in a.binaries:
        if _dest == 'libexpat.so.1' and '/lib/x86_64-linux-gnu/' in _src:
            _new_binaries.append((_dest, _libexpat_path, _typ))
        else:
            _new_binaries.append((_dest, _src, _typ))
    a.binaries = _new_binaries
'''
    return f'''# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('detector', 'detector'), ('ui', 'ui')]
binaries = []
hiddenimports = ['ezdxf.recover']
binaries += collect_dynamic_libs('pyexpat')
tmp_ret = collect_all('ezdxf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

_libexpat_path = {libexpat_str}
a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['scipy', 'plotly', 'cv2', 'sklearn'],
    noarchive=False,
    optimize=0,
)
{libexpat_filter}
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DetectorPlagioCAD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''


def _find_windows_python() -> Path | None:
    candidates = [
        Path(f"~/.wine/drive_c/Python{ver}/python.exe")
        for ver in ["313", "312", "311", "310"]
    ] + [
        Path(f'~/.wine/drive_c/Program Files/Python{ver}/python.exe')
        for ver in ["313", "312", "311", "310"]
    ]
    for p in candidates:
        expanded = p.expanduser()
        if not expanded.exists():
            continue
        try:
            r = subprocess.run(
                ["wine", str(expanded), "-m", "pip", "show", "pyinstaller"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return expanded
        except Exception:
            continue
    return None


def main() -> int:
    current_os = "windows" if sys.platform == "win32" else "linux"

    parser = argparse.ArgumentParser(description="Build executavel do DetectorPlagioCAD")
    parser.add_argument(
        "--target",
        choices=["linux", "windows"],
        default=current_os,
        help=f"SO de destino (padrao: {current_os})",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Caminho do interpretador Python (para compilacao cruzada)",
    )
    args = parser.parse_args()

    cross = args.target != current_os
    python_exe: str | None = args.python

    if cross:
        if python_exe is None:
            if current_os == "linux" and args.target == "windows":
                wp = _find_windows_python()
                if wp:
                    python_exe = str(wp)
                    print(f"Usando Python Windows em: {python_exe}")
                else:
                    print("Compilacao cruzada Linux -> Windows requer um Python Windows.")
                    print("Opcoes:")
                    print("  1) Instale Wine + Python for Windows em ~/.wine/drive_c/Python*")
                    print("  2) Use --python para apontar para o interpretador Windows")
                    return 1
            else:
                print("Compilacao cruzada deste SO requer --python.")
                return 1
    else:
        python_exe = sys.executable

    spec_path = Path(__file__).with_name("DetectorPlagioCAD.spec")
    libexpat = _find_libexpat() if args.target == "linux" else None
    spec_content = _build_spec(libexpat, args.target)
    spec_path.write_text(spec_content, encoding="utf-8")

    cmd: list[str] = [python_exe, "-m", "PyInstaller", "--noconfirm", str(spec_path)]
    if cross and current_os == "linux":
        cmd = ["wine"] + cmd

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
