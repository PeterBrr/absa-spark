"""Sinh thư mục submission/ từ notebooks/ (không sửa bản gốc).

- Điền thông tin nhóm vào ô tiêu đề (thay dòng "**Nhóm X** — _(tên, MSSV các thành viên)_").
- Đặt tên file theo quy định: NhomX_danhsachTen_danhsachMSSV_ABSASpark_NB<k>.ipynb
- Giữ nguyên mọi ô code và output (output chạy trên Kaggle).

Dùng: python tools/make_submission.py   (chạy lại mỗi khi notebook trong notebooks/ thay đổi)
"""
import glob
import os
import sys
import unicodedata

import nbformat

GROUP = 20
MEMBERS = [("Đinh Quang Nhật", "24521248"), ("Võ Tấn Nhã", "24521225")]
PLACEHOLDER = "**Nhóm X** — _(tên, MSSV các thành viên)_"


def ascii_name(name: str) -> str:
    """'Đinh Quang Nhật' → 'DinhQuangNhat' (bỏ dấu, bỏ khoảng trắng)."""
    name = name.replace("Đ", "D").replace("đ", "d")
    plain = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
    return "".join(plain.split())


def submission_name(group: int, members, k: int) -> str:
    names = "-".join(ascii_name(n) for n, _ in members)
    ids = "-".join(mssv for _, mssv in members)
    return f"Nhom{group}_{names}_{ids}_ABSASpark_NB{k}.ipynb"


def make_submission(src_dir: str = "notebooks", out_dir: str = "submission",
                    group: int = GROUP, members=MEMBERS) -> list:
    """Ghi các notebook đã điền tiêu đề vào out_dir; trả về danh sách đường dẫn đã ghi."""
    title = f"**Nhóm {group}** — " + ", ".join(f"{n} ({mssv})" for n, mssv in members)
    sources = sorted(glob.glob(os.path.join(src_dir, "0*.ipynb")))
    notebooks = []
    for k, path in enumerate(sources, start=1):
        nb = nbformat.read(path, as_version=4)
        hits = [c for c in nb.cells if c.cell_type == "markdown" and PLACEHOLDER in c.source]
        if len(hits) != 1:
            raise ValueError(f"{os.path.basename(path)}: cần đúng 1 ô tiêu đề chứa {PLACEHOLDER!r}, thấy {len(hits)}")
        hits[0].source = hits[0].source.replace(PLACEHOLDER, title)
        notebooks.append((k, nb))

    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.ipynb")):  # xóa bản nộp cũ (tên/thứ tự có thể đã đổi)
        os.remove(old)
    written = []
    for k, nb in notebooks:
        dest = os.path.join(out_dir, submission_name(group, members, k))
        nbformat.write(nb, dest)
        written.append(dest)
    return written


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    for path in make_submission():
        print("✓", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
