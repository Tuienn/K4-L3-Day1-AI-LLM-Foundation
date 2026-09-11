# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Ở temperature 0.0 và 0.5, phản hồi có cấu trúc khá ổn định, bám sát yêu cầu và ít biến thể. Khi tăng lên 1.0–1.5, cách chọn sự thật và diễn đạt có xu hướng đa dạng, bất ngờ hơn nhưng cũng dễ lan man hoặc kém nhất quán hơn. Độ trễ không thể hiện quy luật rõ theo temperature vì còn phụ thuộc tải của endpoint/model miễn phí.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Tôi chọn temperature khoảng 0.2 cho chatbot hỗ trợ khách hàng. Mức thấp giúp câu trả lời ổn định, chính xác và nhất quán với chính sách, nhưng vẫn linh hoạt hơn một chút so với 0.0; các thông tin quan trọng vẫn cần được kiểm chứng bằng dữ liệu hoặc công cụ thay vì chỉ dựa vào model.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Workload tạo 10.000 × 3 × 350 = 10,5 triệu output token/ngày. Theo bảng giá trong lab, GPT-4o tốn khoảng 105 USD/ngày còn GPT-4o-mini khoảng 6,30 USD/ngày, nên GPT-4o đắt hơn khoảng 16,7 lần (chênh 98,70 USD/ngày, chưa tính input). GPT-4o xứng đáng cho phân tích tài liệu phức tạp hoặc quyết định cần chất lượng suy luận cao; mini phù hợp cho phân loại, FAQ và các trả lời ngắn có lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Với persona giáo viên tiểu học, câu trả lời ngắn, dùng từ quen thuộc và ví blockchain như một cuốn sổ chung mà mọi người cùng giữ bản sao. Với persona chuyên gia tài chính, phản hồi dài và dùng các thuật ngữ như sổ cái phân tán, cơ chế đồng thuận, mật mã học và tính bất biến. Ví dụ ở bản trẻ em thiên về tình huống đời thường, còn bản chuyên gia tập trung vào giao dịch, rủi ro và kiến trúc hệ thống. Như vậy system prompt định hướng rõ đối tượng, độ sâu, từ vựng và cách tổ chức câu trả lời dù user prompt không đổi.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Với đoạn văn tiếng Việt 114 từ, `count_tokens(..., model="gpt-4o")` cho 149 token, còn ước lượng `114 / 0,75` cho 152 token; chênh lệch khoảng 1,97% so với giá trị ước lượng. Hai cách có thể gần nhau trong một đoạn cụ thể nhưng quy tắc theo số từ chỉ là xấp xỉ. Tiếng Việt có dấu, nhiều âm tiết được viết tách bằng khoảng trắng và một số từ/chuỗi Unicode ít phổ biến trong dữ liệu tokenizer, nên một từ có thể bị tách thành nhiều token hơn tiếng Anh có độ dài tương đương.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi model tạo câu trả lời dài hoặc có độ trễ cao, chẳng hạn chatbot, trợ lý viết và sinh mã, vì người dùng thấy nội dung đầu tiên sớm và biết hệ thống vẫn đang hoạt động. Non-streaming phù hợp hơn khi cần xử lý trọn vẹn kết quả trước khi hiển thị, như phân loại, trả JSON theo schema, kiểm duyệt, tác vụ nền hoặc phản hồi rất ngắn, bởi cách này đơn giản hóa kiểm tra hợp lệ, retry và lưu kết quả.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giãn dần khoảng cách giữa các lần thử, giảm số request trong thời gian dịch vụ quá tải và vẫn cho lỗi ngắn hạn cơ hội phục hồi nhanh. Nếu hàng nghìn client cùng retry sau một delay cố định, chúng có thể thức dậy đồng thời và tạo “thundering herd”, khiến server vừa hồi phục lại quá tải lần nữa. Trong hệ thống thực tế nên cộng thêm jitter ngẫu nhiên vào backoff để phân tán thời điểm retry giữa các client.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Persona tôi chọn là: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn, chính xác bằng tiếng Việt; giải thích thuật ngữ bằng ví dụ đơn giản và nói rõ khi chưa chắc chắn.” Cụm “ngắn gọn, chính xác” giúp phản hồi dễ đọc và hạn chế lan man, còn “nói rõ khi chưa chắc chắn” khuyến khích model thể hiện giới hạn thay vì khẳng định thiếu căn cứ. Chỉ định tiếng Việt giữ trải nghiệm nhất quán với người học trong khóa.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất nên trợ lý nhanh chóng quên mục tiêu hoặc dữ kiện đã nêu ở đầu cuộc trò chuyện. Tôi sẽ thêm bộ nhớ tóm tắt: trước khi cắt các message cũ, gọi một hàm tạo bản tóm tắt ngắn chứa mục tiêu, quyết định và dữ kiện quan trọng, rồi đặt bản tóm tắt đó sau system prompt ở các lượt tiếp theo. Có thể giới hạn token cho bản tóm tắt và cập nhật định kỳ để kiểm soát chi phí.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
