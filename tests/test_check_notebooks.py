import nbformat
import pytest

from tools.check_notebooks import check_notebook, main

COMMON_V2 = "# === COMMON common_cells.py v2 ===\nX = 2\n"
COMMON_V1 = "# === COMMON common_cells.py v1 ===\nX = 1\n"


@pytest.fixture
def common_dir(tmp_path):
    d = tmp_path / "common"
    d.mkdir()
    (d / "common_cells.py").write_text(COMMON_V2, encoding="utf-8")
    return str(d)


def _write_nb(tmp_path, cells, name="nb.ipynb"):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    p = tmp_path / name
    nbformat.write(nb, str(p))
    return str(p)


def _code(src, with_output=True):
    c = nbformat.v4.new_code_cell(src)
    if with_output:
        c.outputs = [nbformat.v4.new_output("stream", name="stdout", text="ok\n")]
    return c


def _good_cells(common=COMMON_V2):
    return [nbformat.v4.new_markdown_cell("## Hướng dẫn chạy\nBật GPU..."),
            _code(common), _code("print(1)")]


def test_ok_notebook_has_no_problems(tmp_path, common_dir):
    assert check_notebook(_write_nb(tmp_path, _good_cells()), common_dir) == []


def test_stale_common_block_detected(tmp_path, common_dir):
    problems = check_notebook(_write_nb(tmp_path, _good_cells(COMMON_V1)), common_dir)
    assert any("common_cells.py" in p for p in problems)


def test_crlf_block_still_matches(tmp_path, common_dir):
    cells = _good_cells(COMMON_V2.replace("\n", "\r\n"))
    assert check_notebook(_write_nb(tmp_path, cells), common_dir) == []


def test_unknown_common_file_detected(tmp_path, common_dir):
    cells = _good_cells() + [_code("# === COMMON khong_co.py v1 ===\n")]
    assert any("khong_co.py" in p for p in check_notebook(_write_nb(tmp_path, cells), common_dir))


def test_hardcoded_kaggle_working_detected(tmp_path, common_dir):
    cells = _good_cells() + [_code("torch.save(m, '/kaggle/working/m.pt')")]
    assert any("/kaggle/working" in p for p in check_notebook(_write_nb(tmp_path, cells), common_dir))


def test_missing_run_guide_detected(tmp_path, common_dir):
    cells = _good_cells()[1:]
    assert any("Hướng dẫn chạy" in p for p in check_notebook(_write_nb(tmp_path, cells), common_dir))


def test_no_outputs_detected(tmp_path, common_dir):
    cells = [_good_cells()[0], _code(COMMON_V2, with_output=False), _code("x", with_output=False)]
    assert any("output" in p for p in check_notebook(_write_nb(tmp_path, cells), common_dir))


def test_main_exit_code(tmp_path, common_dir, capsys):
    good = _write_nb(tmp_path, _good_cells(), "good.ipynb")
    bad = _write_nb(tmp_path, _good_cells()[1:], "bad.ipynb")
    assert main([good, "--common-dir", common_dir]) == 0
    assert main([good, bad, "--common-dir", common_dir]) == 1
    assert "bad.ipynb" in capsys.readouterr().out
