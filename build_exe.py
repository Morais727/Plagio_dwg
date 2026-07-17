import subprocess
import sys

if __name__ == "__main__":
    args = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "DetectorPlagioCAD",
        "--add-data", f"detector{';' if sys.platform == 'win32' else ':'}detector",
        "app.py",
    ]
    subprocess.run(args, check=True)
