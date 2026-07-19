from pathlib import Path

import pytest

from detector.dwg_converter import DwgConverter, DwgConversionError


def test_convert_raises_error_when_file_not_found() -> None:
    converter = DwgConverter()
    with pytest.raises(DwgConversionError, match="Arquivo nao encontrado"):
        converter.convert(Path("nonexistent") / "file.dwg")


def test_is_available_returns_false_when_not_installed() -> None:
    converter = DwgConverter(converter_path="nonexistent_converter")
    assert converter.is_available() is False


def test_convert_raises_error_when_converter_not_available(
    tmp_path: Path,
) -> None:
    converter = DwgConverter(converter_path="definitely_not_available")
    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")
    with pytest.raises(DwgConversionError, match="Conversor nao encontrado"):
        converter.convert(dwg_path)


def test_convert_raises_error_when_converter_fails(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("shutil.which", return_value="fake_converter")
    converter = DwgConverter(converter_path="fake_converter")

    mock_run = mocker.patch("subprocess.run")
    mock_result = mocker.Mock()
    mock_result.returncode = 1
    mock_result.stdout = "error output"
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")

    with pytest.raises(DwgConversionError, match="ODAFileConverter retornou"):
        converter.convert(dwg_path)


def test_convert_cleans_up_temp_dir_on_converter_failure(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("shutil.which", return_value="fake_converter")
    converter = DwgConverter(converter_path="fake_converter")

    mock_run = mocker.patch("subprocess.run")
    mock_result = mocker.Mock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error"
    mock_run.return_value = mock_result

    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")

    with pytest.raises(DwgConversionError):
        converter.convert(dwg_path)


def test_convert_raises_error_when_no_dxf_generated(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("shutil.which", return_value="fake_converter")
    converter = DwgConverter(converter_path="fake_converter")

    mock_run = mocker.patch("subprocess.run")
    mock_result = mocker.Mock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_run.return_value = mock_result

    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")

    with pytest.raises(DwgConversionError, match="DXF nao gerado"):
        converter.convert(dwg_path)


def test_successful_conversion_returns_dxf_path(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("shutil.which", side_effect=lambda x: str(Path("usr") / "bin" / x) if x == "fake_converter" else None)
    converter = DwgConverter(converter_path="fake_converter")

    def fake_run(cmd, **kwargs):
        import subprocess
        output_dir = Path(cmd[2])
        output_dir.mkdir(parents=True, exist_ok=True)
        dxf_file = output_dir / "test.dxf"
        dxf_file.write_text("fake dxf content")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    mock_run = mocker.patch("subprocess.run", side_effect=fake_run)

    dwg_path = tmp_path / "test.dwg"
    dwg_path.write_text("fake dwg content")

    result = converter.convert(dwg_path)

    assert result.name == "test.dxf"
    assert result.exists()
    assert result.read_text() == "fake dxf content"
