# Nguồn dữ liệu

- Dataset: **ABSA Vietnamese** — https://www.kaggle.com/datasets/cthng123/absa-vietnamese
- Tác giả: cthng123 (Kaggle)
- Giấy phép: **CC BY 4.0** (Attribution 4.0 International) — được phép chia sẻ lại, cần ghi nguồn.
- Nội dung: ~11.700 bình luận về giày trên Shopee, 8 khía cạnh, 4 nhãn.

Bản sao trong thư mục này giữ nguyên file gốc, không chỉnh sửa. Notebook tải từ đây
(qua `raw.githubusercontent.com`) khi chạy ngoài Kaggle.

| File | Số dòng |
|---|---|
| `train_data.csv` | 8.424 |
| `val_data.csv` | 936 |
| `test_data.csv` | 2.340 |

Cột: `Review` (văn bản), `Price, Shipping, Outlook, Quality, Size, Shop_Service, General, Others`.
Nhãn: `-1 = None` (không nhắc đến), `0 = Negative`, `1 = Positive`, `2 = Neutral`.
Không có cột độ tin cậy của nhãn.
