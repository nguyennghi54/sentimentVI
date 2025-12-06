# Trợ lý phân loại cảm xúc tiếng Việt sử dụng Transformer 

## Cấu trúc thư mục

SentimentVI/

├── 📄 app.py            # Mã nguồn cho main program

├── 📄 requirements.txt  # Danh sách các thư viện cần thiết

├── 📄 sentiment_history.db      # Cơ sở dữ liệu SQLite (Có thể xóa, tự sinh ra khi chạy app)

└── 📄 README.md         # Hướng dẫn sử dụng

## Tính năng Chính

Lõi MLP: Mô hình PhoBERT-base đã được fine-tuned https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment, tích hợp qua Hugging Face pipeline để đảm bảo độ chính xác cao trong việc xử lý ngữ cảnh Tiếng Việt.

Chuẩn hóa Input: Tự động làm sạch và chuẩn hóa các từ viết tắt/từ lóng Tiếng Việt (ko, dc, wa) trước khi phân tích.

Lưu trữ Lịch sử: Sử dụng SQLite3 để lưu trữ lịch sử các lần phân tích vào file sentiment_history.db.

Giao diện Web: Triển khai qua Streamlit Cloud, cung cấp giao diện trực quan và dễ sử dụng.

## Hướng dẫn chạy app

A. Truy cập qua Web
Truy cập vào đường link URL của ứng dụng đã được triển khai trên Streamlit Cloud: https://@sentimentvi.streamlit.app/ (bỏ @)

B. Chạy cục bộ

1. Clone repo này về
  
2. pip install -r requirements.txt
  
3. streamlit run test1.py

## Hướng dẫn sử dụng

1. Phân tích Cảm xúc
   
Nhập liệu: Nhập câu văn bản Tiếng Việt vào vùng "Nhập câu tiếng việt:".

Kiểm tra Input: Ứng dụng sẽ cảnh báo nếu câu quá ngắn (dưới 5 ký tự) hoặc quá dài (trên 50 ký tự).

Thực hiện: Nhấn nút "Phân tích".

Kết quả: Kết quả sẽ hiển thị ngay lập tức dưới mục "Kết quả phân tích:".

- Xanh lá (POSITIVE): Cảm xúc Tích cực.

- Đỏ (NEGATIVE): Cảm xúc Tiêu cực.

- Xanh dương (NEUTRAL): Cảm xúc Trung lập.

2. Quản lý Lịch sử
   
Lưu trữ: Mỗi lần phân tích thành công, bản ghi sẽ được tự động lưu vào file database cục bộ (sentiment_history.db).

Xem Lịch sử: Nhấn vào mục mở rộng "Lịch sử phân tích" để hiển thị 50 kết quả gần nhất, bao gồm Văn bản đã chuẩn hóa và Nhãn dự đoán.

Lưu ý: Do ứng dụng được deploy tạm trên Streamlit Share Cloud, khi app khởi động, file .db được copy và ghi vào hệ thống tệp tin tạm. Nếu app bị reboot hoặc ngủ (do không sử dụng quá lâu), toàn bộ file sẽ bị xóa và Streamlit tải lại file .db gốc từ github. Streamlit không tự đẩy code lên github được, vậy nên dữ liệu chỉ được lưu trong mỗi phiên làm việc tạm thời.
