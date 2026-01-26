# Vietnamese Scam Scenario Dataset (29 Classes)

## 📌 Giới thiệu

Dataset này được xây dựng nhằm phục vụ cho các bài toán:
- Phát hiện và phân loại kịch bản lừa đảo (Scam Scenario Classification)
- Phát hiện dấu hiệu lừa đảo trong hội thoại (Fraud Signal Detection)
- Nghiên cứu hành vi và quy trình lừa đảo qua điện thoại, SMS, mạng xã hội
- Huấn luyện và đánh giá các mô hình NLP / LLM cho tiếng Việt

Dataset tập trung vào **29 nhóm kịch bản lừa đảo phổ biến tại Việt Nam**, được tổng hợp từ các nguồn thực tế, dữ liệu dịch thuật và dữ liệu sinh tổng hợp có kiểm soát.

---

## 🧠 Danh sách 29 nhóm kịch bản lừa đảo

1. **Lừa đảo đặt bàn ăn**  
   Giả mạo dịch vụ đặt bàn nhà hàng, tạo áp lực thời gian và yêu cầu chuyển tiền cọc.

2. **Lừa đảo đặt phòng khách sạn**  
   Giả mạo nhân viên khách sạn hoặc nền tảng đặt phòng, thông báo thanh toán nhầm và yêu cầu cung cấp thông tin thẻ/OTP.

3. **Lừa đảo định danh CCCD**  
   Giả mạo cơ quan công an, thông báo trùng thông tin với tội phạm, yêu cầu định danh lại để chiếm đoạt tài khoản.

4. **Lừa đảo giả cơ quan nhà nước (GTVT, TTTT...)**  
   Giả danh cán bộ điều tra liên quan đến vụ án để thu thập thông tin cá nhân và OTP.

5. **Lừa đảo giả danh cơ quan thuế**  
   Thông báo vi phạm thuế, hóa đơn bất hợp pháp, yêu cầu cài ứng dụng giả mạo.

6. **Lừa đảo giả danh công an**  
   Thông báo lệnh triệu tập, yêu cầu chuyển tiền hoặc tài sản để “bảo quản”.

7. **Lừa đảo giả mạo thương hiệu, tổ chức (Điện lực, viễn thông...)**  
   Thông báo nợ cước, nợ tiền điện và yêu cầu truy cập link giả mạo.

8. **Lừa đảo giả sàn thương mại điện tử**  
   Thông báo đơn hàng bất thường, yêu cầu cài ứng dụng để chiếm quyền thiết bị.

9. **Lừa đảo giao đơn hàng không có thực**  
   Giả mạo nhân viên sàn TMĐT, yêu cầu OTP để hủy đơn hàng ảo.

10. **Lừa đảo hải quan nộp phạt**  
    Thông báo bưu kiện chứa hàng cấm, yêu cầu nộp phạt để tránh bị bắt.

11. **Lừa đảo khóa SIM**  
    Thông báo SIM vi phạm, yêu cầu nộp phí chuẩn hóa.

12. **Lừa đảo máy lọc nước**  
    Giả mạo nhân viên bảo trì, bán sản phẩm kém chất lượng với giá cao.

13. **Lừa đảo mở, rút tiền thẻ tín dụng**  
    Giả danh nhân viên ngân hàng, yêu cầu thông tin thẻ để rút tiền trái phép.

14. **Lừa đảo PCCC**  
    Giả mạo cán bộ PCCC, đe dọa xử phạt và chào mời khóa học/chứng chỉ giả.

15. **Lừa đảo sử dụng trang web giả mạo**  
    Dẫn nạn nhân vào website giả để thu thập thông tin đăng nhập.

16. **Lừa đảo ứng dụng ví điện tử / ví trả sau**  
    Giả mạo MoMo, ZaloPay… để đánh cắp tiền.

17. **Thông báo nhận tiền hỗ trợ, trợ cấp an sinh xã hội**  
    Dẫn nạn nhân vào web giả mạo để chiếm đoạt thông tin cá nhân.

18. **Lừa đảo vay vốn**  
    Mời chào vay lãi suất thấp, yêu cầu đóng phí trước.

19. **Lừa đảo cài ứng dụng độc hại**  
    Dụ dỗ cài app chứa mã độc để tự động rút tiền.

20. **Lừa đảo lợi dụng sự cố mất an toàn thông tin**  
    Giả mạo CIC/ngân hàng thông báo tài khoản có vấn đề sau rò rỉ dữ liệu.

21. **Lừa đảo báo lỗi tài khoản**  
    Chủ động khóa tài khoản, sau đó gọi điện hỗ trợ để chiếm quyền thiết bị.

22. **Lừa đảo từ thiện**  
    Giả mạo tổ chức uy tín kêu gọi quyên góp.

23. **Lừa đảo mạo danh người thân quen**  
    Báo tin khẩn cấp để vay tiền.

24. **Lừa đảo lấy lại tiền treo, tiền bị lừa**  
    Giả danh luật sư/công an, hứa giúp lấy lại tiền đã mất.

25. **Lừa đảo đe dọa sự việc không có thực**  
    Sử dụng deepfake, ảnh/video nhạy cảm để tống tiền.

26. **Lừa đảo trúng tuyển, tuyển sinh**  
    Giả mạo thông báo trúng tuyển việc làm hoặc nhập học.

27. **Lừa đảo thanh toán tiền giao hàng**  
    Giả mạo shipper, yêu cầu thanh toán hoặc truy cập link độc hại.

28. **Lừa đảo đầu tư tiền ảo**  
    Dẫn dụ đầu tư dự án tiền ảo giả mạo với lợi nhuận phi thực tế.

29. **Lừa đảo mạo danh bác sĩ, cơ sở y tế**  
    Bán thuốc giả, thu tiền khám chữa bệnh không có thật.

---

## 🔎 Cấu trúc chung của một kịch bản lừa đảo

Mỗi kịch bản trong dataset thường tuân theo chuỗi hành vi sau:

1. **Xác lập danh tính & tiếp cận**  
   Giả mạo tổ chức/cá nhân đáng tin cậy để tạo niềm tin ban đầu.

2. **Tạo áp lực / sợ hãi / khẩn cấp**  
   Đe dọa hậu quả pháp lý, tài chính hoặc cảm xúc để thúc ép nạn nhân.

3. **Thu thập thông tin cá nhân**  
   Yêu cầu CCCD, thông tin ngân hàng, mã OTP, thông tin đăng nhập.

4. **Yêu cầu hành động nguy hiểm**  
   Chuyển tiền, cung cấp OTP, cài đặt ứng dụng, truy cập link giả mạo.

---


