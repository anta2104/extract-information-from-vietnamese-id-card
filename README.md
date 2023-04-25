# extract-information-from-vietnamese-id-card

Bước 1 : tìm 4 góc của căn cước (hoặc 3 góc sau đó nội suy ra góc thứ 4 ) 
Bước 2 : cắt ảnh theo 4 góc của căn cước đã tìm được nhằm mục đích làm cho mô hình dễ nhận diện và độ chính xác cao hơn 
Bước 3 : tìm các vùng thông tin trên căn cước theo nhãn đã dán 
Bước 4 : dùng thư viện vietOCR để chuyển từ dạng image-->text

Mô hình phát hiện các góc : 
Mô hình phát hiện các vùng chữ : 
Tool gán nhãn : labelImg 
