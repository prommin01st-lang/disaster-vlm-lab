set -e
# Task 13: merge LoRA → GGUF (Q8_0/Q4_K_M + mmproj F16) → HF private — CPU runtime พอ
# llama.cpp ปักที่ b10909 = รุ่นเดียวกับไบนารี Vulkan ที่ใช้รันบนเครื่อง (format ตรงกันแน่นอน)
LLAMA_TAG=b10909
free -g | head -2; nproc; df -h /content | tail -1
git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp /content/llama.cpp
echo "llama.cpp $(git -C /content/llama.cpp rev-parse HEAD)"
pip install -q /content/llama.cpp/gguf-py sentencepiece
mkdir -p /content/llama-bin
curl -sfL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz" \
  | tar -xz -C /content/llama-bin
QUANT=$(find /content/llama-bin -name llama-quantize -type f | head -1)
export LD_LIBRARY_PATH="$(dirname "$QUANT"):${LD_LIBRARY_PATH:-}"
"$QUANT" --help 2>&1 | head -1 || true
LLAMA_DIR=/content/llama.cpp LLAMA_QUANTIZE="$QUANT" python scripts/export_gguf.py
