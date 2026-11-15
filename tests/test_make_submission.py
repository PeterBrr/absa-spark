import nbformat
import pytest

from tools.make_submission import PLACEHOLDER, ascii_name, make_submission, submission_name

MEMBERS = [("Đinh Quang Nhật", "24521248"), ("Võ Tấn Nhã", "24521225")]


def test_ascii_name_handles_vietnamese_d():
    assert ascii_name("Đinh Quang Nhật") == "DinhQuangNhat"
    assert ascii_name("Võ Tấn Nhã") == "VoTanNha"


def test_submission_name():
    assert submission_name(20, MEMBERS, 3) == "Nhom20_DinhQuangNhat-VoTanNha_24521248-24521225_ABSASpark_NB3.ipynb"


def _nb(path, title_line=PLACEHOLDER):
    nb = nbformat.v4.new_notebook()
    code = nbformat.v4.new_code_cell("print(1)")
    code.outputs = [nbformat.v4.new_output("stream", name="stdout", text="1\n")]
    nb.cells = [nbformat.v4.new_markdown_cell("## Hướng dẫn chạy"),
                nbformat.v4.new_markdown_cell(f"# NB — tiêu đề\n\n{title_line}\n\nMục tiêu"), code]
    nbformat.write(nb, str(path))


def test_make_submission_fills_title_and_keeps_outputs(tmp_path):
    src, out = tmp_path / "notebooks", tmp_path / "submission"
    src.mkdir()
    for name in ["01_a.ipynb", "02_b.ipynb"]:
        _nb(src / name)
    written = make_submission(str(src), str(out), 20, MEMBERS)
    assert [p.split("_")[-1] for p in written] == ["NB1.ipynb", "NB2.ipynb"]
    nb = nbformat.read(written[0], 4)
    title = nb.cells[1].source
    assert PLACEHOLDER not in title
    assert "**Nhóm 20** — Đinh Quang Nhật (24521248), Võ Tấn Nhã (24521225)" in title
    assert nb.cells[2].outputs[0]["text"] == "1\n"  # output Kaggle giữ nguyên
    assert nbformat.read(str(src / "01_a.ipynb"), 4).cells[1].source.count(PLACEHOLDER) == 1  # không sửa bản gốc


def test_make_submission_fails_without_placeholder(tmp_path):
    src = tmp_path / "notebooks"
    src.mkdir()
    _nb(src / "01_a.ipynb", title_line="**Nhóm 7** — ai đó")
    with pytest.raises(ValueError, match="01_a.ipynb"):
        make_submission(str(src), str(tmp_path / "submission"), 20, MEMBERS)


def test_make_submission_removes_stale_files(tmp_path):
    src, out = tmp_path / "notebooks", tmp_path / "submission"
    src.mkdir()
    out.mkdir()
    (out / "Nhom20_cu_NB9.ipynb").write_text("{}")
    _nb(src / "01_a.ipynb")
    make_submission(str(src), str(out), 20, MEMBERS)
    assert sorted(p.name for p in out.glob("*.ipynb")) == [submission_name(20, MEMBERS, 1)]
