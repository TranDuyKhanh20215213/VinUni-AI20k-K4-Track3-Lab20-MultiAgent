# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

### Trả lời (dựa trên `reports/benchmark_report.md` của repo này)

**1. Nên dùng multi-agent khi:** câu hỏi cần **evidence có nguồn trích dẫn được** và có thể tách
rõ thành các bước retrieval → phân tích → viết. Trong benchmark thực tế của repo này (query về
kiến trúc multi-agent), pipeline Researcher → Analyst → Writer đạt **citation coverage 100%** và
quality score 10.0/10, so với baseline không trích dẫn được nguồn nào (citation coverage rỗng,
quality 7.5/10) — vì baseline không có bước retrieval nên không có gì để cite, chỉ "nhớ" lại từ
tri thức nội tại của model. Multi-agent đáng giá khi: (a) câu trả lời sai/không có nguồn là rủi ro
cao (báo cáo kỹ thuật, tài liệu nội bộ cần audit được), (b) có một corpus/nguồn cụ thể phải bám
sát thay vì để model tự bịa, (c) task đủ phức tạp để tách biệt "tìm gì" (Researcher) khỏi "đánh giá
độ tin cậy" (Analyst) khỏi "viết sao cho rõ" (Writer) là có ý nghĩa thật, không chỉ chia cho có.

**2. Không nên dùng multi-agent khi:** câu hỏi ngắn, câu trả lời có thể lấy trực tiếp từ tri thức
sẵn có của model, không cần trích dẫn, và độ trễ/chi phí quan trọng hơn độ chắc chắn của nguồn.
Cùng một query, multi-agent trong benchmark **chậm hơn baseline ~9.3s (6.87s → 16.16s)** và **tốn
gấp ~3 lần chi phí token** (`$0.0003` → `$0.0009`) vì phải chạy 3 lượt LLM call thay vì 1, cộng
thêm bước retrieval. Nếu task không có nhu cầu grounding thật sự (ví dụ: giải thích khái niệm phổ
thông, brainstorm nhanh, câu hỏi mà baseline đã trả lời "đủ tốt"), phần chi phí/độ trễ thêm vào
không đổi lại được lợi ích tương xứng — giống đúng điều `docs/design_template.md` mục "Why
multi-agent?" đã nêu: multi-agent chỉ đáng khi task decomposition tạo ra nhu cầu thông tin/kiểm
chứng thực sự khác nhau giữa các bước, nếu không thì overhead điều phối sẽ xóa sạch lợi ích chất
lượng.
