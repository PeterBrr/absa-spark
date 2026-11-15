"""Kiểm tra notebook trước khi commit/nộp.

Quy tắc:
  1. Ô đầu tiên là Markdown bắt đầu bằng "## Hướng dẫn chạy".
  2. Mọi ô code bắt đầu bằng "# === COMMON <file> v<N> ===" phải giống hệt common/<file>.
  3. Không viết cứng đường dẫn "/kaggle/working".
  4. Có ít nhất một ô code có output (thầy cần thấy kết quả).

Dùng: python tools/check_notebooks.py notebooks/*.ipynb [--common-dir common]
"""
import argparse
import glob
import json
import os
import re
import sys

HEADER_RE = re.compile(r"^# === COMMON (\S+) v(\d+) ===")


def _source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def check_notebook(nb_path: str, common_dir: str = "common") -> list:
    """Trả về danh sách lỗi (rỗng nghĩa là đạt)."""
    with open(nb_path, encoding="utf-8") as fh:
        cells = json.load(fh).get("cells", [])
    problems = []

    first = cells[0] if cells else {}
    if first.get("cell_type") != "markdown" or not _source(first).lstrip().startswith("## Hướng dẫn chạy"):
        problems.append("Ô đầu tiên phải là Markdown '## Hướng dẫn chạy'")

    for i, cell in enumerate(cells):
        src = _source(cell)
        if "/kaggle/working" in src:
            problems.append(f"Ô {i}: viết cứng đường dẫn /kaggle/working (dùng ./models/)")
        if cell.get("cell_type") != "code":
            continue
        m = HEADER_RE.match(src.lstrip())
        if not m:
            continue
        fname, version = m.group(1), m.group(2)
        path = os.path.join(common_dir, fname)
        if not os.path.isfile(path):
            problems.append(f"Ô {i}: khối {fname} không có trong {common_dir}/")
            continue
        with open(path, encoding="utf-8") as fh:
            expected = fh.read()
        if _norm(src) != _norm(expected):
            exp_ver = HEADER_RE.match(expected.lstrip())
            exp_ver = exp_ver.group(2) if exp_ver else "?"
            problems.append(f"Ô {i}: khối {fname} v{version} khác bản trong {common_dir}/ (v{exp_ver}) — dán lại")

    if not any(c.get("cell_type") == "code" and c.get("outputs") for c in cells):
        problems.append("Notebook chưa có output nào — hãy Run All trước khi commit/nộp")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--common-dir", default="common")
    args = parser.parse_args(argv)
    try:  # Windows: tránh UnicodeEncodeError (✓ ✗, tiếng Việt) khi output bị chuyển hướng/pipe
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    failed = False
    notebooks = []
    for pattern in args.notebooks:  # cmd/PowerShell không tự mở rộng *.ipynb → tự mở rộng
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            print(f"✗ {pattern}: không có file nào khớp")
            failed = True
        notebooks.extend(matches)
    for nb in notebooks:
        problems = check_notebook(nb, args.common_dir)
        if problems:
            failed = True
            print(f"✗ {nb}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"✓ {nb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
