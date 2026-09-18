# Sentiment Analysis with LSTM — IMDb Reviews

## 1. Tổng quan đề bài

| | |
|---|---|
| **Goal** | Phân loại cảm xúc (positive/negative) trong review phim |
| **Dataset** | IMDb Reviews — 50,000 review (25,000 train / 25,000 test), cân bằng tuyệt đối 50/50 |
| **Model** | Embedding + LSTM + Dense (Linear) |
| **Task Type** | Binary Classification |
| **Extension** | GRU, Bidirectional LSTM, GloVe pre-trained embeddings |
| **Môi trường** | Google Colab (GPU T4), PyTorch, artifact lưu trên Google Drive |

## 2. Pipeline thực hiện — 5 Notebook

| Notebook | Nội dung chính |
|---|---|
| **Stage1** | Load IMDb, EDA cơ bản + nâng cao (negation, duplicate, OOV), xây vocabulary, tokenize, padding, chia train/val/test, tạo DataLoader |
| **Stage2** | Baseline `Embedding → LSTM → Dense → Sigmoid`, model summary, train với early stopping |
| **Stage3** | Đánh giá chi tiết baseline: ROC/AUC, Precision-Recall curve, phân tích lỗi theo độ dài review, top ca dự đoán sai |
| **Stage4** | Huấn luyện 3 extension (GRU, Bidirectional LSTM, GloVe+LSTM) cùng điều kiện để so sánh công bằng |
| **Stage5** | So sánh tổng hợp 4 model, ví dụ dự đoán thực tế |

Artifact (vocab, tensors, checkpoint, kết quả) được lưu bền vững trên Google Drive (`MyDrive/imdb_sentiment_artifacts/`) giữa các giai đoạn — mỗi notebook có thể chạy độc lập, không cần giữ session của notebook trước.

## 3. Nguồn gốc & Cách thu thập dữ liệu

**Nguồn:** Large Movie Review Dataset (Stanford, Maas et al. 2011, công bố tại ACL 2011).

**Quy trình thu thập gốc:**
- Review **crawl trực tiếp từ IMDb.com**, kèm điểm số 1–10 do người dùng chấm.
- Gán nhãn dựa trên điểm số có sẵn: **≤4/10 → negative**, **≥7/10 → positive**; loại bỏ hoàn toàn điểm 5–6/10 (trung tính) để tránh nhãn mơ hồ.
- Giới hạn **tối đa 30 review/phim** để tránh phim nổi tiếng làm lệch phân phối.
- Train/test **không trùng phim**, mỗi tập cân bằng tuyệt đối 50% positive/negative.

**Cách truy cập trong project:** load qua Hugging Face `datasets` (`load_dataset("stanfordnlp/imdb")`) — bản mirror chính thức, giữ nguyên đặc điểm thu thập gốc.

## 4. EDA — Kết quả chi tiết

### 4.1 Thống kê cơ bản

| Chỉ số | Train | Test |
|---|---:|---:|
| Số lượng mẫu | 25,000 | 25,000 |
| Tỷ lệ positive | 50% | 50% |
| Độ dài trung bình (từ) | 233.8 | 228.5 |
| Độ dài median (từ) | 174 | ~174 |
| Độ dài min–max | 10 – 2,470 | tương tự |

- Độ dài trung bình theo nhãn: **negative 230.9 từ**, **positive 236.7 từ** — positive dài hơn một chút.

### 4.2 Phân tích từ vựng & ngôn ngữ

| Phân tích | Kết quả |
|---|---|
| Top từ phổ biến theo nhãn (sau khi loại stopword) | Trích xuất riêng cho positive/negative — cho thấy từ vựng đặc trưng cảm xúc |
| Wordcloud positive vs negative | Trực quan hóa sự khác biệt từ vựng giữa 2 lớp |
| **Tỷ lệ review chứa từ phủ định** | **Positive: 79.7%** — **Negative: 89.8%** |
| Top bigram theo nhãn | Cho thấy cụm từ 2-gram đặc trưng, ví dụ các cụm phủ định kết hợp tính từ |

→ Negative review dùng phủ định nhiều hơn đáng kể — lý do bài toán cần model hiểu **ngữ cảnh chuỗi** (LSTM/GRU) thay vì chỉ đếm từ khóa riêng lẻ (bag-of-words), vì phủ định đảo ngược nghĩa của từ theo sau ("not good" ≠ "good").

### 4.3 Kiểm tra chất lượng dữ liệu

| Kiểm tra | Kết quả | Ý nghĩa |
|---|---|---|
| Review trùng lặp hoàn toàn trong train | 96/25,000 (0.38%) | Không đáng kể |
| **Review trùng giữa train và test** | **123 review (0.49%)** | **Data leakage có thật** trong dataset gốc — khiến accuracy đo được có thể nhỉnh hơn thực tế đôi chút |
| Tỷ lệ review còn chứa thẻ HTML `<br />` | **58.7%** | Xác nhận bước làm sạch text là bắt buộc trước khi tokenize |

### 4.4 Phân tích OOV (Out-of-Vocabulary)

| Vocab size | Tỷ lệ OOV ước tính |
|---:|---:|
| 10,000 | 5.91% |
| **20,000 (đã chọn)** | **2.81%** |
| 30,000 | 1.69% |
| 40,000 | 1.07% |

→ Vocab 20,000 từ là điểm cân bằng hợp lý: OOV đủ thấp, không làm embedding layer phình to không cần thiết (lợi ích biên giảm dần khi tăng vocab thêm).

## 5. Tiền xử lý dữ liệu (Preprocessing)

1. **Làm sạch text:** bỏ thẻ `<br />`, chuyển lowercase, loại ký tự đặc biệt (regex `[^a-z0-9'\s]`).
2. **Tokenize:** tách từ theo khoảng trắng sau khi làm sạch.
3. **Xây vocabulary:** đếm tần suất từ trên toàn bộ train set, giữ lại **20,000 từ phổ biến nhất**, thêm 2 token đặc biệt `<pad>` (index 0) và `<unk>` (index 1).
4. **Text → sequence số nguyên:** map mỗi từ sang index trong vocab, từ hiếm/lạ → `<unk>`.
5. **Padding/Truncating:** cố định độ dài mỗi review về **MAX_LEN = 250 token** (cắt bớt nếu dài hơn, đệm số 0 nếu ngắn hơn).
6. **Chia tập:** train 90% / validation 10% (tách ngẫu nhiên từ 25,000 train, seed=42), test giữ nguyên 25,000 mẫu riêng biệt.
7. **DataLoader:** batch size 64, shuffle=True cho train, False cho val/test.

## 6. Kiến trúc Model

### 6.1 Sơ đồ luồng dữ liệu

```
Text → Tokenize → Embedding → LSTM/GRU/BiLSTM → Dense(Linear) → Sigmoid → Positive/Negative
```

### 6.2 Chi tiết từng biến thể

| Biến thể | Embedding | RNN Layer | Đầu ra RNN → Dense |
|---|---|---|---|
| **Baseline** | Học từ đầu, dim=128 | LSTM 1 chiều, hidden=128 | hidden_dim (128) → 1 |
| **GRU** | Học từ đầu, dim=128 | GRU 1 chiều, hidden=128 | hidden_dim (128) → 1 |
| **Bidirectional LSTM** | Học từ đầu, dim=128 | LSTM 2 chiều, hidden=128 | hidden_dim×2 (256, nối 2 chiều) → 1 |
| **GloVe+LSTM** | Pre-trained GloVe 6B.100d, dim=100, fine-tune | LSTM 1 chiều, hidden=128 | hidden_dim (128) → 1 |

Tất cả đều dùng `torch.nn.utils.rnn.pack_padded_sequence` để RNN bỏ qua hoàn toàn token đệm (`<pad>`) khi tính hidden state, và `Dropout(0.3)` trước tầng Dense.

### 6.3 Siêu tham số huấn luyện (chung cho cả 4 model)

| Tham số | Giá trị |
|---|---|
| Vocab size | 20,000 |
| Max sequence length | 250 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | Binary Cross-Entropy (`BCEWithLogitsLoss`) |
| Dropout | 0.3 |
| Gradient clipping | `max_norm = 5.0` |
| Epoch tối đa | 15 |
| Early stopping | patience = 3 (dựa trên validation loss) |
| Checkpoint | Lưu theo validation accuracy tốt nhất |

### 6.4 Số tham số & thời gian train

| Model | Tổng tham số | Epoch dừng thực tế | Train time (giây) |
|---|---:|---:|---:|
| Baseline | 2,692,225 | 10 | – |
| GRU | 2,659,201 | 7 | 41.5 |
| Bidirectional LSTM | 2,824,449 | 8 | 66.8 |
| GloVe+LSTM | 2,117,889 (ít nhất, nhờ embedding 100d thay vì 128d) | 6 | 40.2 |

## 7. Vấn đề kỹ thuật gặp phải & Cách khắc phục

### 7.1 Lỗi #1 — Padding làm nhiễu hidden state

**Triệu chứng:** lần train đầu tiên (chưa dùng `pack_padded_sequence`), test accuracy chỉ **~52–58%** (gần đoán ngẫu nhiên), train accuracy dao động bất thường giữa các epoch (ví dụ: 50.5% → 54.2% → 63.7% → 57.2% → 62.7%, không tăng đều).

**Nguyên nhân:** RNN đọc toàn bộ chuỗi 250 token kể cả phần đệm `<pad>` ở cuối các review ngắn hơn — hidden state cuối cùng bị "loãng" bởi hàng chục/hàng trăm token vô nghĩa thay vì phản ánh đúng nội dung.

**Khắc phục:**
- Thêm `torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)` trước khi đưa vào RNN — tính độ dài thật của mỗi review bằng `(x != pad_idx).sum(dim=1)`.
- Thêm gradient clipping (`clip_grad_norm_`, max_norm=5.0) để ổn định quá trình học.

**Kết quả sau fix:** train accuracy tăng đều 62% → 88% qua các epoch (baseline), test accuracy đạt ~84%.

### 7.2 Lỗi #2 — Số epoch cố định (5) chưa đủ hội tụ

**Triệu chứng:** với 5 epoch cố định, Bidirectional LSTM cho kết quả bất thường: Accuracy 81.83%, **Recall chỉ 0.717** (Precision 0.899) — model thiên lệch nặng về dự đoán "negative", trong khi val loss của các model khác vẫn đang giảm ở epoch cuối cùng.

**Chẩn đoán:** không phải lỗi kiến trúc — chỉ là **chưa train đủ lâu để hội tụ**, đặc biệt với BiLSTM có gấp đôi tham số ở tầng cuối (do nối 2 chiều) nên cần nhiều epoch hơn.

**Khắc phục:** tăng epoch tối đa lên 15, thêm **early stopping** (dừng khi val loss không cải thiện sau 3 epoch liên tiếp).

**Kết quả sau fix (BiLSTM):**
| Chỉ số | 5 epoch cố định | Early stopping (dừng epoch 8) | Thay đổi |
|---|---:|---:|---:|
| Accuracy | 81.83% | 84.25% | +2.42 điểm |
| Precision | 0.899 | 0.816 | -0.083 |
| Recall | 0.717 | 0.884 | **+0.167** |
| F1-score | 0.798 | 0.849 | +0.051 |

→ Xác nhận: kiến trúc phức tạp hơn chỉ cần thời gian huấn luyện phù hợp, không phải "kém hơn về bản chất".

## 8. Đánh giá chi tiết Baseline (Stage3)

Ngoài Accuracy/Precision/Recall/F1/Confusion Matrix cơ bản, Stage3 phân tích sâu thêm:

### 8.1 ROC Curve & Precision-Recall Curve

| Chỉ số | Giá trị |
|---|---:|
| **AUC (Area Under ROC Curve)** | **0.9108** |
| **Average Precision (AP)** | **0.9032** |

→ Model phân biệt tốt giữa 2 lớp ở hầu hết các ngưỡng threshold, không chỉ tốt riêng tại ngưỡng mặc định 0.5.

### 8.2 Accuracy theo độ dài review — Phát hiện quan trọng

| Độ dài (số token) | Accuracy |
|---|---:|
| 0–50 | 87.7% |
| 51–100 | 87.7% |
| 101–150 | 87.5% |
| 151–200 | 86.1% |
| **201–250 (bị truncate ở MAX_LEN)** | **80.4%** |

→ **`MAX_LEN=250` đang cắt mất thông tin quan trọng** của các review dài, làm giảm accuracy rõ rệt (giảm ~7 điểm so với accuracy tổng thể) ở nhóm review bị cắt. Đây là giới hạn thiết kế cụ thể, có thể khắc phục bằng cách tăng MAX_LEN hoặc dùng chiến lược truncate thông minh hơn (ví dụ giữ đầu + cuối thay vì chỉ giữ đầu).

### 8.3 Phân tích ca dự đoán sai với độ tự tin cao nhất

Các ca model dự đoán sai nhưng với xác suất rất tự tin (gần 0 hoặc gần 1) thường có đặc điểm:
- Review pha trộn cảm xúc (vừa khen vừa chê trong cùng đoạn văn)
- Yếu tố hoài niệm/so sánh phức tạp (ví dụ so sánh phiên bản cũ/mới của một series)
- Chứa nhiều từ hiếm bị thay bằng `<unk>`, làm mất thông tin ngữ nghĩa quan trọng

## 9. Kết quả so sánh cuối cùng — 4 Model

| Model | Params | Train time (s) | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Embedding+LSTM+Dense) | 2,692,225 | – | 0.8439 | 0.8468 | 0.8398 | 0.8433 |
| **GRU** | 2,659,201 | 41.5 | **0.8643** | 0.8584 | 0.8726 | **0.8654** |
| Bidirectional LSTM | 2,824,449 | 66.8 | 0.8350 | 0.8449 | 0.8206 | 0.8326 |
| GloVe + LSTM | 2,117,889 | 40.2 | 0.8553 | 0.8820 | 0.8204 | 0.8501 |

*(GloVe: 19,154/20,000 từ vocab tìm thấy trong GloVe 6B.100d — 95.8% coverage)*

### So sánh với lần chạy trước đó (kiểm tra độ ổn định)

| Model | Lần chạy 1 (Accuracy) | Lần chạy 2 — chính thức (Accuracy) |
|---|---:|---:|
| Baseline | 84.26% | 84.39% |
| GRU | 85.35% | **86.43%** |
| Bidirectional LSTM | 84.25% | 83.50% |
| GloVe+LSTM | **85.76%** | 85.53% |

→ **Thứ hạng giữa GRU và GloVe+LSTM đổi chỗ giữa 2 lần chạy** (lần 1: GloVe+LSTM nhất; lần 2: GRU nhất) do random initialization trọng số model không cố định seed. Chênh lệch giữa 2 model này (dưới 1 điểm %) nằm trong biên độ nhiễu ngẫu nhiên — **kết luận hợp lý là 2 kiến trúc này có hiệu quả gần như tương đương** trên bài toán này, không nên khẳng định một model vượt trội tuyệt đối chỉ dựa trên 1 lần chạy.

### Nhận xét tổng hợp

- **GRU và GloVe+LSTM** là 2 lựa chọn hiệu quả nhất, đồng thời có **chi phí tính toán thấp nhất** (thời gian train nhanh nhất, dừng sớm nhất nhờ early stopping).
- **Bidirectional LSTM** luôn là model **tốn thời gian train nhất** (66.8s, gần gấp đôi GRU/GloVe) do phải tính toán 2 chiều, nhưng kết quả không vượt trội hơn — trade-off không đáng giá cho bài toán này ở quy mô hiện tại.
- **Baseline** vẫn là điểm tham chiếu hợp lý — cải thiện từ baseline lên GRU/GloVe là có ý nghĩa (~1-2 điểm % accuracy) nhưng không quá lớn, cho thấy kiến trúc LSTM cơ bản đã nắm bắt được phần lớn tín hiệu quan trọng trong dữ liệu.

## 10. Ví dụ dự đoán thực tế

Model tốt nhất không chỉ bắt được từ khóa cảm xúc rõ ràng ("terrible", "masterpiece") mà còn nhận diện đúng các câu **phê bình nhẹ nhàng, gián tiếp, không dùng từ ngữ tiêu cực rõ ràng**. Ví dụ:

> "...features the usual car chases, fights... all of this is entertaining and competently handled but there is nothing that really blows you away..."

- **Actual:** negative — **Predicted:** negative — **Xác suất:** 0.003 (rất tự tin, đúng)

Đây là minh chứng model học được sắc thái ngữ nghĩa ở mức câu, không chỉ dựa vào từ khóa đơn lẻ — nhất quán với phát hiện ở mục 8.3 rằng các ca dự đoán sai thường liên quan đến review phức tạp/pha trộn cảm xúc hơn là review đơn giản, rõ ràng.

## 11. Ý nghĩa của các đặc điểm dữ liệu đối với kết quả model

- **Cân bằng 50/50** ngay từ khâu thu thập → accuracy là chỉ số đáng tin cậy cho bài toán này (không lo lệch lớp). Đây cũng là lý do khi model gặp lỗi kỹ thuật (mục 7.1), accuracy rơi đúng về ngưỡng ~50-58% — dấu hiệu rõ ràng "không học được gì" chứ không phải do dữ liệu lệch.
- **Loại bỏ điểm trung tính (5-6/10)** khi thu thập giúp bài toán "rõ ràng" hơn nhưng đồng nghĩa model chỉ học phân biệt cảm xúc **cực đoan** (rất thích/rất ghét) — chưa chắc tổng quát tốt cho đánh giá pha trộn/trung lập trong thực tế.
- **Giới hạn 30 review/phim** khi thu thập đảm bảo model học **pattern ngôn ngữ chung** của sentiment, không phải "học thuộc" theo phim cụ thể — lý giải vì sao 84-86% accuracy có ý nghĩa thực chất.
- **Độ dài không đồng đều** ảnh hưởng đến model theo 2 tầng: gây lỗi kỹ thuật ban đầu (đã sửa ở mục 7.1) VÀ gây mất thông tin thiết kế qua MAX_LEN (phát hiện ở mục 8.2, chưa khắc phục).
- **Tỷ lệ phủ định cao** (79.7-89.8%) giải thích vì sao kiến trúc tuần tự (LSTM/GRU/BiLSTM) phù hợp hơn các phương pháp bag-of-words — thứ tự từ (đặc biệt là phủ định) mang ý nghĩa quyết định.
- **123 review trùng train/test** là giới hạn của bản thân dataset gốc, khiến accuracy đo được có thể nhỉnh hơn thực tế đôi chút — cần nêu rõ như một giới hạn của phương pháp đánh giá, không phải lỗi của project.

## 12. Cấu trúc file

```
imdb_sentiment_artifacts/          (Google Drive)
├── vocab.pkl                      # Vocabulary (20,000 từ)
├── imdb_processed.pt              # Tensors đã tokenize + pad
├── baseline_lstm_best.pt          # Checkpoint baseline
├── baseline_results.json          # Kết quả baseline
├── gru_best.pt / bilstm_best.pt / glove_lstm_best.pt   # Checkpoint extensions
├── extension_results.json         # Kết quả 3 extension
└── glove/glove.6B.100d.txt        # Pre-trained GloVe embeddings

Notebooks (Colab):
├── Stage1.ipynb   — Data prep + EDA chi tiết
├── Stage2.ipynb   — Baseline training
├── Stage3.ipynb   — Đánh giá chi tiết baseline
├── Stage4.ipynb   — Extensions (GRU, BiLSTM, GloVe)
└── Stage5.ipynb   — So sánh tổng hợp
```

## 13. Kết luận

Project đã hoàn thành đầy đủ và vượt yêu cầu đề bài: xây dựng baseline Embedding+LSTM+Dense, thử nghiệm 3 hướng mở rộng (GRU, Bidirectional LSTM, GloVe embeddings), kèm EDA và đánh giá chuyên sâu (bigram/negation, OOV, ROC/AUC, phân tích lỗi theo độ dài). Kết quả cho thấy GRU và GloVe+LSTM là 2 lựa chọn hiệu quả nhất và gần như tương đương, trong khi Bidirectional LSTM tốn chi phí tính toán nhất nhưng không mang lại lợi ích tương xứng.

Quá trình debug thực tế (lỗi padding, nhu cầu early stopping) cùng các phát hiện từ EDA sâu (data leakage 123 review trùng lặp, MAX_LEN cắt mất thông tin ở review dài) là những đóng góp quan trọng, cho thấy hiểu biết không chỉ dừng ở việc chạy được model mà còn ở việc diễn giải đúng hành vi và giới hạn của nó.

**Hướng phát triển tiếp theo:**
- Tăng `MAX_LEN` (ví dụ 350-400) để giảm mất thông tin ở review dài, dựa trên phát hiện ở mục 8.2.
- Cố định random seed cho việc khởi tạo trọng số model để so sánh công bằng hơn giữa các lần chạy.
- Loại bỏ 123 review trùng lặp train/test để đảm bảo đánh giá không bị lạc quan giả tạo.
- Thử nghiệm attention mechanism trên baseline/GRU để xem có cải thiện khả năng xử lý review dài hay không.
