# Requirement Specification — DVL Incident Classifier

ระบบจำแนกภาพแจ้งเหตุด้วย VLM (fine-tune `Qwen3.5-2B`) + HTTP API + หน้าเว็บลองเล่น สำหรับเป็นผู้ช่วยคัดกรองของระบบแจ้งเหตุ SOS

| | |
|---|---|
| เวอร์ชันเอกสาร | 1.0 (2026-10-09) — เขียนจากระบบที่สร้างและวัดผลแล้ว (as-built) |
| ขอบเขตโค้ด | repo นี้: `dvl/`, `scripts/`, `api/`, `docker/`, `docker-compose.yml`, `bruno/` |
| เอกสารที่เกี่ยวข้อง | [`REPORT.md`](../REPORT.md) (ผลประเมินเต็ม) · [`README.md`](../README.md) (วิธีรัน) · [`bruno/`](../bruno/) (API contract + ชุดเทส) |

**สถานะในตาราง:** ✅ ผ่าน (มีหลักฐานวัดจริง) · ⚠️ ผ่านบางส่วน / มีเงื่อนไข · ❌ ไม่ผ่าน · ⬜ ยังไม่ได้ทดสอบ
ทุกข้อมี ID เพื่ออ้างอิงตอนเขียน test case และ traceability

---

## 1. บทนำ

### 1.1 วัตถุประสงค์

ผู้แจ้งเหตุในระบบ SOS แนบภาพมาพร้อมเรื่องแจ้ง ระบบนี้ช่วยเจ้าหน้าที่ด้วยการ:

1. เดาประเภทเหตุจากภาพตามคลังประเภทของ SOS เพื่อ**เติมฟอร์มล่วงหน้า**ให้เจ้าหน้าที่ยืนยัน
2. ชี้ภาพที่**ควรให้คนดู** (`needs_review`) โดยเฉพาะภาพที่โมเดลคิดว่า "ไม่ใช่เหตุ" แต่ไม่มั่นใจพอ

ระบบ**ไม่ใช่**ตัวตัดสินสั่งการ — ทุกผลต้องมีเจ้าหน้าที่เป็นผู้ตัดสินสุดท้าย

### 1.2 ผู้เกี่ยวข้อง

| บทบาท | ความต้องการหลัก |
|---|---|
| เจ้าหน้าที่รับแจ้ง (อบต./ศูนย์รับแจ้ง) | ได้ประเภทเหตุเบื้องต้นเร็ว ๆ และรู้ว่าภาพไหนต้องดูเอง |
| ทีมพัฒนา backend ของ SOS (ASP.NET Core) | API ที่ contract ชัด เสถียร เรียกง่าย และมี error code ที่จัดการได้ |
| ผู้ดูแลระบบ / ผู้ติดตั้ง | ติดตั้งบนเครื่องที่มี GPU เล็ก ๆ หรือ Docker ได้ และรู้ว่าต้องใช้ทรัพยากรเท่าไร |
| ผู้พัฒนาโมเดล | วัดผลซ้ำได้ เปลี่ยนโมเดล/threshold แล้วรู้ผลกระทบ |

### 1.3 คำศัพท์

| คำ | ความหมาย |
|---|---|
| `incident_type` | key ประเภทเหตุ ตรงกับ SOS incident catalog (เช่น `flood`) |
| `no_incident` | ค่าของโมเดลเท่านั้น (ไม่มีใน SOS) = ภาพไม่ใช่เหตุ |
| miss | ภาพเหตุจริงที่ถูกตอบว่า `no_incident` — ความผิดพลาดที่อันตรายที่สุด |
| false alarm | ภาพที่ไม่ใช่เหตุแต่ถูกตอบว่าเป็นเหตุ |
| abstain gate | กติกาเปลี่ยน `no_incident` ที่ไม่มั่นใจเป็น `unsure` + `needs_review` |
| `p_incident` | 1 − P(`category` = `null`) จาก logprob ของ token — ความน่าจะเป็นที่เป็นเหตุ |
| T | threshold ของ gate (ค่าปัจจุบัน 3.2e-5) |
| gold | 247 ภาพจาก test ที่คนตรวจ label แล้ว |

---

## 2. ขอบเขต

**ในขอบเขต:** ภาพนิ่ง 1 ภาพต่อคำขอ · 12 ประเภทเหตุที่ดูจากภาพได้ + `unsure` + `no_incident` · HTTP API บนเครื่องเดียว ·
หน้าเว็บลองเล่น · Docker (cpu/cuda) · โมเดล GGUF Q4_K_M ผ่าน llama.cpp

**นอกขอบเขต:** วิดีโอ/หลายภาพต่อเรื่อง · ระบุตำแหน่ง จำนวนผู้บาดเจ็บ หรือระดับความเร่งด่วน 2–4 ของ catalog ·
อ่านข้อความในภาพ · ประเภทเหตุอื่นใน catalog ที่ดูจากภาพไม่ได้ · ยืนยันตัวตนผู้ใช้หลายคน/สิทธิ์ราย role ·
การเก็บภาพหรือประวัติฝั่งเซิร์ฟเวอร์ · การ scale หลายเครื่อง

---

## 3. Business requirements

| ID | ความต้องการ | วัดอย่างไร | สถานะ |
|---|---|---|---|
| BR-01 | ลดภาระเจ้าหน้าที่ในการเลือกประเภทเหตุ | macro-F1 สูงกว่า base model ทั้ง gold และ test | ✅ gold 0.525 → 0.697 · test 0.321 → 0.607 |
| BR-02 | ไม่ทำให้เหตุจริงตกหล่นมากกว่าการไม่ใช้ระบบ | miss ของระบบ (Q4 + gate) ≤ miss ของ base | ✅ gold 0.018 = 0.018 · test¹ 0.003 < 0.007 |
| BR-03 | ไม่เพิ่มภาระด้วยเหตุเท็จ | false alarm ≤ base | ✅ gold 0.400 → 0.080 · test 0.484 → 0.117–0.130 |
| BR-04 | รันได้บนเครื่องของหน่วยงานที่มี GPU เล็ก | ใช้ VRAM ≤ 4 GB บนการ์ด 6 GB | ✅ 2.2 GB (llama-server) |
| BR-05 | ใช้งานได้ตามกฎหมาย/ไลเซนส์สำหรับหน่วยงานรัฐ | ข้อมูลเทรนอนุญาตให้ใช้งานจริง | ❌ ข้อมูลส่วนใหญ่ CC BY-NC-SA — ต้องทบทวน/เทรนใหม่ก่อนใช้จริง (CON-03) |
| BR-06 | ใช้กับภาพแจ้งเหตุในประเทศไทยได้ | ผลบนชุดทดสอบภาพไทย | ⬜ ยังไม่มีชุดทดสอบภาพไทย |

¹ test ที่ตัด gold ออก (2,997 ภาพ) ดู REPORT.md 5.3

---

## 4. Functional requirements

### 4.1 โมเดล (FR-M)

| ID | ความต้องการ | สถานะ / หลักฐาน |
|---|---|---|
| FR-M-01 | รับภาพ 1 ภาพ ตอบ JSON บรรทัดเดียว `{"category","incident_type","severity"}` เรียง key ตายตัว | ✅ |
| FR-M-02 | `incident_type` ต้องเป็นหนึ่งใน 14 ค่า: `forest_fire`, `building_fire`, `vehicle_fire`, `smoke_detected`, `flood`, `storm`, `landslide`, `damaged_structure`, `road_hazard`, `building_collapse`, `water_rescue`, `vehicle_collision`, `unsure`, `no_incident` | ✅ `dvl/catalog.py` |
| FR-M-03 | `category` ต้องตรงกับ `incident_type` ตาม catalog และเป็น `null` เมื่อ `no_incident` | ✅ parser บังคับ (`dvl/schema.py`) |
| FR-M-04 | `severity` ∈ {`none`,`mild`,`severe`} โดย `none` ใช้กับ `no_incident` เท่านั้น | ✅ parser ซ่อมคู่ที่ขัดกัน |
| FR-M-05 | ถ้าโมเดลตอบผิดรูปแบบ ระบบไม่ crash แต่คืน `unsure` + `valid=false` | ✅ `json_valid_rate` 1.000 บน gold/test |
| FR-M-06 | `confidence` คำนวณจาก logprob ของ token ค่า `incident_type` — โมเดลไม่เขียนตัวเลขเอง | ✅ `dvl/confidence.py` |
| FR-M-07 | ภาพถูกเตรียมแบบเดียวกับตอนเทรน: หมุนตาม EXIF, RGBA → พื้นขาว, ด้านยาว ≤ 512 px | ✅ `dvl/imgutil.py` |
| FR-M-08 | system prompt อยู่ที่เดียว (`dvl/prompt.py`) และเหมือนกันทุกที่ที่ใช้ (เทรน, predict, API) | ✅ |
| FR-M-09 | ไม่เปิด thinking — คำตอบต้องไม่มี `<think>` | ✅ ตรวจใน `predict_gguf.py` |

### 4.2 Abstain gate (FR-G)

| ID | ความต้องการ | สถานะ / หลักฐาน |
|---|---|---|
| FR-G-01 | ถ้าโมเดลตอบ `no_incident` และ `p_incident ≥ T` → ผลสุดท้ายเป็น `incident_type=unsure`, `category=other`, `severity=mild`, `needs_review=true`, `review_reason=uncertain_no_incident` | ✅ `dvl/llamacpp.py`, `tests/test_api.py` |
| FR-G-02 | `p_incident` = 1 − ผลรวมความน่าจะเป็นของ token `null` ใน top-5 ที่ตำแหน่งค่า `category`; ถ้า `null` ไม่อยู่ใน top-5 ถือว่า `p_incident` = 1 (ส่งให้คนดู) | ✅ |
| FR-G-03 | T ต้องเลือกบนชุด val (ไม่ใช่ชุดที่ใช้รายงานผล) เป็นค่า T ใหญ่สุดที่ miss บน val ≤ 1% | ✅ `scripts/abstain_sweep.py` → T = 3.2e-5 |
| FR-G-04 | ตั้ง T ได้จาก config (`DVL_ABSTAIN_T`) โดยไม่ต้องแก้โค้ด | ✅ |
| FR-G-05 | เมื่อเปลี่ยนไฟล์โมเดล, quant, เวอร์ชัน llama.cpp หรือ backend (Vulkan/CUDA/CPU) ต้อง sweep T ใหม่ | ⚠️ เป็นขั้นตอนปฏิบัติ — Docker ใช้ b10902 และ CPU/CUDA ยังไม่ได้ sweep |

### 4.3 HTTP API (FR-A)

| ID | ความต้องการ | สถานะ / หลักฐาน |
|---|---|---|
| FR-A-01 | `POST /v1/classify` รับ multipart field `image` (jpeg/png/webp) แล้วตอบ JSON ตาม 6.2 | ✅ |
| FR-A-02 | `GET /v1/labels` คืนรายการ 14 ประเภทพร้อม `category` และชื่อไทย | ✅ |
| FR-A-03 | `GET /health` ตอบ 200 เมื่อ llama-server พร้อม และ 503 เมื่อไม่พร้อม — ไม่ต้องใช้ key | ✅ |
| FR-A-04 | `needs_review=true` เมื่อ gate ทำงาน, โมเดลตอบ `unsure` เอง หรือคำตอบผิดรูปแบบ พร้อม `review_reason` ตาม 6.3 | ✅ |
| FR-A-05 | คืน `model_incident_type` (คำตอบดิบก่อน gate; `null` ถ้าคำตอบผิดรูปแบบ) | ✅ |
| FR-A-06 | ตรวจทั้ง Content-Type และเนื้อไฟล์จริง — ไฟล์ที่ไม่ใช่ jpeg/png/webp ได้ 415 | ✅ |
| FR-A-07 | error ทุกตัวตอบ `{"error": <code>, "message": <text>}` พร้อม HTTP status ตาม 6.4 | ✅ |
| FR-A-08 | ส่งภาพต่อให้ llama-server เป็น PNG (lossless) ไม่ encode JPEG ซ้ำ | ✅ ตรงกับสคริปต์ 240/247 (JPEG ได้ 234/247) |
| FR-A-09 | ถ้าตั้ง `DVL_API_KEY` ทุก `/v1/*` ต้องมี header `X-API-Key` ที่ถูกต้อง (เทียบแบบ constant-time) มิฉะนั้น 401 | ✅ |
| FR-A-10 | เรียก llama-server ทีละคำขอ (คิว) เพราะเปิด `-np 1` | ✅ |
| FR-A-11 | มีเอกสาร OpenAPI อัตโนมัติ (`/docs`) และ Bruno collection ที่รันเป็นชุดเทสได้ | ✅ Bruno 23/23 (ไม่มี key), 25/25 (มี key) |

### 4.4 หน้าเว็บลองเล่น (FR-W)

| ID | ความต้องการ | สถานะ |
|---|---|---|
| FR-W-01 | เปิดที่ `GET /` ของ API (origin เดียวกัน ไม่ต้องตั้ง CORS) | ✅ |
| FR-W-02 | ส่งภาพได้ 3 ทาง: คลิกเลือก, ลากวาง, Ctrl+V | ✅ |
| FR-W-03 | แสดงผลเป็น 3 สถานะชัดเจน: พบเหตุ (แดง) / ไม่ใช่เหตุ (เขียว) / ควรให้เจ้าหน้าที่ตรวจ (เหลือง + เหตุผลภาษาไทย) | ✅ |
| FR-W-04 | แสดงรายละเอียด: `incident_type`, หมวด, ความรุนแรง, confidence, `p_incident` เทียบ T, เวลา, JSON ดิบ | ✅ |
| FR-W-05 | เมื่อ gate เปลี่ยนคำตอบ ต้องแสดงคำตอบเดิมของโมเดล และระบุว่า confidence เป็นของคำตอบเดิม | ✅ |
| FR-W-06 | แสดงสถานะ llama-server (ตรวจ `/health` ทุก 15 วินาที) | ✅ |
| FR-W-07 | เก็บประวัติภาพที่ลองล่าสุด 12 ภาพในหน้า (หายเมื่อ refresh) กดดูผลย้อนหลังได้ | ✅ |
| FR-W-08 | ใส่ API key ได้ เก็บใน localStorage ของเบราว์เซอร์ และทำงานได้แม้ localStorage ใช้ไม่ได้ | ✅ |
| FR-W-09 | ตรวจชนิด/ขนาดไฟล์ฝั่งเบราว์เซอร์ก่อนส่ง และแปล error code เป็นข้อความไทย | ✅ |
| FR-W-10 | ใช้งานได้บนจอมือถือ (กว้าง 390 px) และรองรับ dark mode | ✅ |

### 4.5 Deployment (FR-D)

| ID | ความต้องการ | สถานะ |
|---|---|---|
| FR-D-01 | `api/run.sh` เปิด llama-server (ถ้ายังไม่มี) + API ด้วยคำสั่งเดียว และปิด llama-server ที่ตัวเองเปิดเมื่อจบ | ⚠️ ปิดถูกเมื่อ Ctrl+C แต่ถ้าปิด terminal (SIGHUP) llama-server ค้างได้ |
| FR-D-02 | Docker Compose profile `cpu` และ `cuda` เลือกผ่าน `COMPOSE_PROFILES` | ✅ cpu ทดสอบเต็ม · ⬜ cuda ตรวจแค่ config |
| FR-D-03 | service `models` ดาวน์โหลด GGUF จาก HF (private) เมื่อยังไม่มีไฟล์ และตรวจ sha256; token ไม่ถูก bake ลง image | ✅ |
| FR-D-04 | API container รันแบบ non-root, มี HEALTHCHECK, publish เฉพาะพอร์ต API | ✅ uid 10001 |
| FR-D-05 | ค่าทั้งหมดตั้งผ่าน env (`.env.example`) | ✅ |

---

## 5. Non-functional requirements

### 5.1 คุณภาพของผล (NFR-Q) — วัดบน GGUF Q4_K_M + gate

| ID | เกณฑ์ | ค่าที่วัดได้ | สถานะ |
|---|---|---|---|
| NFR-Q-01 | miss บน gold ≤ base (0.018) | 0.018 (4/222) | ✅ |
| NFR-Q-02 | miss บน test ที่ไม่รวม gold ≤ 1% | 0.003 | ✅ |
| NFR-Q-03 | false alarm บน gold ≤ 0.20 | 0.080 | ✅ |
| NFR-Q-04 | macro-F1 บน gold ≥ 0.65 | 0.697 | ✅ |
| NFR-Q-05 | สัดส่วนที่ส่งให้คนดู ≤ 15% | gold 5.3% · test¹ 10.4% | ✅ |
| NFR-Q-06 | JSON ถูกรูปแบบ ≥ 99% | 100% | ✅ |
| NFR-Q-07 | ทุกคลาสที่มี ≥ 10 ภาพใน test ได้ F1 ≥ 0.5 (bf16) | ต่ำสุด building_collapse 0.58 | ✅ |
| NFR-Q-08 | คลาสหายาก (`smoke_detected`, `unsure`) ได้ F1 > 0 | 0.00 | ❌ ข้อมูลไม่พอ (train 34 / 20 ภาพ) |
| NFR-Q-09 | calibration: ECE ≤ 0.10 | 0.158 (gold, Q4) | ❌ อย่าใช้ `confidence` แทน `needs_review` |

### 5.2 ประสิทธิภาพและทรัพยากร (NFR-P) — RTX 2060 6GB, llama.cpp b10909 Vulkan

| ID | เกณฑ์ | ค่าที่วัดได้ | สถานะ |
|---|---|---|---|
| NFR-P-01 | เวลาตอบต่อภาพ p95 ≤ 1.0 วินาที (ไม่มีคิว) | p50 0.50 · p95 0.56 · max 0.93 วินาที | ✅ |
| NFR-P-02 | throughput ≥ 1 ภาพ/วินาที | ~1.9 ภาพ/วินาที (60 คำขอต่อเนื่อง 31.5 วินาที) | ✅ |
| NFR-P-03 | VRAM ≤ 4 GB | llama-server 2.2 GB | ✅ |
| NFR-P-04 | RAM ≤ 4 GB รวมทุก process | llama-server 1.4 GB + API 80 MB | ✅ |
| NFR-P-05 | ดิสก์สำหรับโมเดล ≤ 3 GB | Q4_K_M 1.2 GB + mmproj 0.64 GB | ✅ |
| NFR-P-06 | ทำงานได้บน CPU อย่างเดียว (โหมดสำรอง) | 3–5 วินาที/ภาพ ผ่าน Docker cpu | ✅ ช้ากว่า GPU ~8 เท่า |
| NFR-P-07 | ผลเหมือนเดิมเมื่อส่งภาพเดิมซ้ำ | ไม่ deterministic 100% (ต่าง ~1 ภาพ/247 ข้ามรอบ) | ⚠️ |

### 5.3 ความปลอดภัย (NFR-S)

| ID | เกณฑ์ | สถานะ |
|---|---|---|
| NFR-S-01 | bind ที่ 127.0.0.1 เป็นค่าเริ่มต้น; ถ้าเปิดให้เครื่องอื่นต้องตั้ง `DVL_API_KEY` | ✅ |
| NFR-S-02 | จำกัดขนาด body 10 MB (+64 KB overhead) โดยนับระหว่างสตรีม — รวมคำขอแบบ chunked ที่ไม่มี Content-Length → 413 `too_large` | ✅ ทดสอบส่ง 30 MB / 200 MB |
| NFR-S-03 | กัน decompression bomb: ตรวจขนาดจาก header ก่อนถอดภาพ เกิน 40 ล้านพิกเซล → 413 `too_many_pixels` | ✅ ทดสอบ PNG 69 byte ประกาศ 20000×20000 |
| NFR-S-04 | ถอดภาพพร้อมกันไม่เกิน 2 ภาพ | ✅ |
| NFR-S-05 | ไม่เก็บภาพหรือผลลัพธ์ลงดิสก์ฝั่งเซิร์ฟเวอร์ | ✅ |
| NFR-S-06 | หน้าเว็บมี CSP `default-src 'self'`, `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` และไม่ใส่ข้อมูลจาก API ลง DOM แบบ HTML | ✅ |
| NFR-S-07 | ไม่มี secret/token ใน repo หรือ Docker image | ✅ ตรวจก่อน publish |
| NFR-S-08 | timeout ต่อคำขอไปที่ llama-server (`DVL_TIMEOUT`, ค่าเริ่มต้น 60 วินาที) → 504 | ✅ · ⚠️ เวลารอคิวไม่นับรวม — ฝั่ง SOS ต้องตั้ง HttpClient timeout ให้นานกว่า |

### 5.4 การดูแลรักษาและทดสอบ (NFR-M)

| ID | เกณฑ์ | สถานะ |
|---|---|---|
| NFR-M-01 | unit test ไม่ต้องใช้ GPU (llama-server ถูก mock) | ✅ 113 tests |
| NFR-M-02 | วัดผลซ้ำได้ด้วยสคริปต์ (`evaluate.py`, `abstain_sweep.py`) และบันทึกผลใน `reports/` | ✅ |
| NFR-M-03 | Bruno collection ข้ามคำขอที่ต้องใช้ภาพจาก dataset ได้เองเมื่อไม่มีภาพ | ✅ |

---

## 6. Interface contract

### 6.1 Request

```
POST /v1/classify
Content-Type: multipart/form-data
X-API-Key: <key>            # เฉพาะเมื่อเปิด DVL_API_KEY
image=<ไฟล์ jpeg|png|webp ≤ 10 MB, ≤ 40 ล้านพิกเซล>
```

### 6.2 Response 200

| field | type | ความหมาย |
|---|---|---|
| `category` | string \| null | หมวดตาม catalog; `null` เมื่อ `no_incident` |
| `incident_type` | string | ผลสุดท้าย (หลัง gate) — หนึ่งใน 14 ค่าของ FR-M-02 |
| `severity` | `none`\|`mild`\|`severe` | ความรุนแรงที่เห็นในภาพ (ไม่ใช่ระดับ 2–4 ของ catalog) |
| `confidence` | number 0–1 | ความน่าจะเป็นของคำตอบ**เดิม**ของโมเดล |
| `p_incident` | number 0–1 | 1 − P(category = null) |
| `needs_review` | boolean | ควรให้เจ้าหน้าที่ดู |
| `review_reason` | string \| null | ดู 6.3 |
| `model_incident_type` | string \| null | คำตอบดิบก่อน gate |
| `incident_type_name_th` | string \| null | ชื่อไทยของ `incident_type` |
| `valid` | boolean | โมเดลตอบถูกรูปแบบหรือไม่ |
| `model` | string | ชื่อโมเดล (`DVL_MODEL_NAME`) |
| `threshold` | number | T ที่ใช้ |
| `latency_ms` | integer | เวลาประมวลผลรวม (ไม่รวมเวลาส่งไฟล์) |

ตัวอย่าง:

```json
{"category": "other", "incident_type": "unsure", "severity": "mild", "confidence": 1.0,
 "p_incident": 0.000116, "needs_review": true, "review_reason": "uncertain_no_incident",
 "model_incident_type": "no_incident", "incident_type_name_th": "ไม่แน่ใจประเภทเหตุ", "valid": true,
 "model": "dvl-qwen3.5-2b-Q4_K_M", "threshold": 3.2e-05, "latency_ms": 781}
```

### 6.3 `review_reason`

| ค่า | เมื่อไร | ความหมายต่อเจ้าหน้าที่ |
|---|---|---|
| `uncertain_no_incident` | gate ทำงาน (FR-G-01) | อาจเป็นเหตุ — **ห้ามทิ้ง** |
| `model_unsure` | โมเดลตอบ `unsure` เอง | เป็นเหตุแต่ไม่รู้ประเภท |
| `invalid_output` | โมเดลตอบผิดรูปแบบ | ไม่มีข้อมูลจากโมเดล ให้ดูเอง |
| `null` | — | ไม่ต้องดูเป็นพิเศษ (ยังต้องยืนยันตามขั้นตอนปกติ) |

### 6.4 Error

| HTTP | `error` | เมื่อไร |
|---|---|---|
| 400 | `missing_image` | ไม่มี field `image` |
| 400 | `undecodable_image` | ถอดภาพไม่ได้ |
| 401 | `unauthorized` | key ผิดหรือไม่ส่ง (เมื่อเปิด `DVL_API_KEY`) |
| 413 | `too_large` | body เกิน 10 MB |
| 413 | `too_many_pixels` | ภาพเกิน 40 ล้านพิกเซล |
| 415 | `unsupported_media_type` | ไม่ใช่ jpeg/png/webp (ตรวจทั้ง Content-Type และเนื้อไฟล์) |
| 502 | `llama_unreachable` / `llama_error` | ติดต่อ llama-server ไม่ได้ หรือได้คำตอบผิดปกติ |
| 504 | `llama_timeout` | llama-server ไม่ตอบภายใน `DVL_TIMEOUT` |

### 6.5 Configuration

| env | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `DVL_LLAMA_URL` | `http://127.0.0.1:8091` | ที่อยู่ llama-server |
| `DVL_ABSTAIN_T` | `3.2e-5` | threshold ของ gate |
| `DVL_MODEL_NAME` | `dvl-qwen3.5-2b-Q4_K_M` | ชื่อที่คืนใน response |
| `DVL_API_KEY` | (ว่าง = ปิด) | key สำหรับ `/v1/*` |
| `DVL_TIMEOUT` | `60` | วินาทีที่รอ llama-server ต่อคำขอ |
| `DVL_API_BIND` / `DVL_API_PORT` | `127.0.0.1` / `8092` | (Docker) ที่อยู่ที่ publish |

---

## 7. ข้อมูลและโมเดล (DR)

| ID | ความต้องการ | สถานะ |
|---|---|---|
| DR-01 | แบ่ง train/val/test แบบไม่ให้ภาพเกือบซ้ำรั่วข้ามชุด (aHash, Hamming ≤ 4) | ✅ |
| DR-02 | val/test คงสัดส่วนธรรมชาติ (~74% `no_incident`); balance เฉพาะ train | ✅ |
| DR-03 | มีชุด gold ที่คนตรวจ ≥ 25 ภาพต่อคลาส (ถ้ามีพอ) | ⚠️ 247 ภาพ; คลาสหายากมี 2–5 ภาพ; ผู้ตรวจคนเดียวและเห็น label เดิม |
| DR-04 | ไม่แจกจ่ายภาพจาก dataset ใน repo สาธารณะ | ✅ ลบออกจากประวัติ git แล้ว |
| DR-05 | โมเดลเก็บบน HF แบบ private: adapter `Petanque/dvl-qwen35-2b-lora`, GGUF `Petanque/dvl-qwen3.5-2b-gguf` | ✅ |

---

## 8. ข้อจำกัดและสมมติฐาน (CON)

| ID | รายละเอียด |
|---|---|
| CON-01 | ระบบเป็นผู้ช่วย ไม่ใช่ผู้ตัดสิน — ผล `no_incident` ที่ `needs_review=false` ยังพลาดได้ ~0.3–2% |
| CON-02 | ไม่มีภาพจากประเทศไทยในข้อมูลเทรน/ทดสอบ |
| CON-03 | ข้อมูลเทรนส่วนใหญ่เป็น CC BY-NC-SA 4.0 (CrisisMMD) และบางชุดไม่ระบุไลเซนส์ — ต้องทบทวนก่อนใช้ในหน่วยงานจริง |
| CON-04 | T ผูกกับไฟล์ Q4_K_M + llama.cpp b10909 + Vulkan + การส่งภาพแบบ PNG |
| CON-05 | ประมวลผลทีละคำขอ (`-np 1`) — ถ้าคำขอเข้ามาพร้อมกันจะต่อคิว |
| CON-06 | `confidence` อิ่มที่ ~1.0 เพราะโมเดลตัดสินใจที่ token ของ `category` ก่อน — ใช้ `needs_review` แทน |

---

## 9. Acceptance criteria ก่อนต่อเข้าระบบ SOS จริง

ระบบรุ่นนี้**ผ่าน**เกณฑ์สำหรับทดลองใช้ภายใน (BR-01–04, NFR-Q-01–06, NFR-P, NFR-S) แต่**ยังไม่ผ่าน**สำหรับใช้งานจริง จนกว่าจะครบ:

1. BR-05: ทบทวนไลเซนส์ หรือเทรนใหม่ด้วยข้อมูลที่ใช้ได้
2. BR-06: ทดสอบกับภาพแจ้งเหตุจริงจากไทย ≥ 300 ภาพ โดยผู้ตรวจ 2 คนที่ไม่เห็น label เดิม แล้ว miss ≤ 1% และส่งให้คนดู ≤ 15%
3. FR-G-05: sweep T ใหม่บน backend ที่ใช้จริง (เช่น Docker CUDA)
4. FR-D-01: แก้ `run.sh` ให้ปิด llama-server เมื่อได้ SIGHUP หรือใช้ Docker/systemd แทน
5. ตกลงขั้นตอนงานกับเจ้าหน้าที่: ภาพ `needs_review` เข้าคิวไหน และไม่มีเรื่องใดถูกปิดจากผลโมเดลอย่างเดียว

---

## 10. Traceability

| ความต้องการ | ทดสอบที่ |
|---|---|
| FR-M-01–06 | `tests/test_schema.py`, `tests/test_confidence.py`, `tests/test_catalog.py` |
| FR-G-01–02 | `tests/test_llamacpp.py`, `tests/test_api.py` |
| FR-A-01–10, NFR-S-02–04 | `tests/test_api.py`, `bruno/03-Classify/*` |
| FR-A-03, FR-A-02 | `bruno/01-Health`, `bruno/02-Labels` |
| FR-W-01 | `tests/test_api.py::test_web_ui_served_without_key` |
| NFR-Q-* | `reports/*-gold.json`, `reports/*-test.json`, `reports/abstain-q4.md` |
| NFR-P-* | วัดด้วยมือบน RTX 2060 (REPORT.md 5.4) — ยังไม่มีสคริปต์ benchmark อัตโนมัติ |
