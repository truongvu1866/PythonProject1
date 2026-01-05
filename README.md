# Hướng dẫn dùng và test demo  
Lưu ý terminal sẽ mở từ đầu đến khi không muốn test nữa  
**B1**: tải toàn bộ tài nguyên [tại dây](https://drive.google.com/drive/folders/1VOX4pZpq7kbXU321qEYmJkvwgvk7I_2z?usp=sharing)  
**B2**: tạo môi trường ảo của cho project (nếu dùng ide thì không cần làm)  
* **B2.1** vào bên trong folder vừa tải về nhấp chuột trái vào khoảng trống chọn **Open in terminal**  
* **B2.2** Thêm folder .venv chứa các tài nguyên môi trường ảo   
```bash 
python -m venv .venv
```
* **B2.3** Cài đặt các package cần thiết để chạy code    
```bash
pip install -r requirements.txt
```
(yêu cầu đã có python và pip trên máy, nếu chưa có yêu cầu tự tìm và cài đặt)  
**B3** Tạo thêm user cho database (có thể bỏ qua nếu chỉ muốn test dưới dạng unkown)  
* **B3.1** Vào folder data tạo thêm một folder với tên người dùng mới  
* **B3.2** Thêm ảnh của người dùng mới vào folder vừa tạo  
* **B3.3** Gõ lệnh sau:  
```bash
python build_database.py
```
xuất hiện [OK] trên terminal là đã xong  
**B4** Chạy chương trình:  
```bash
python main.py
```
