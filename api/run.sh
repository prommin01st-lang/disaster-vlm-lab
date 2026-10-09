#!/usr/bin/env bash
# เปิด llama-server (ถ้ายังไม่ healthy) แล้วรัน uvicorn api.app:app — Ctrl+C / exit ปิด llama-server ที่สคริปต์นี้เปิดเอง
# override ได้ด้วย env: DVL_MODEL, DVL_MMPROJ, DVL_LLAMA_BIN, DVL_LLAMA_PORT, DVL_API_HOST, DVL_API_PORT, DVL_PYTHON
# (ตัวแปรของแอปเอง เช่น DVL_API_KEY / DVL_ABSTAIN_T / DVL_TIMEOUT ส่งต่อให้ uvicorn ตามปกติ)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${DVL_MODEL:-models/gguf/dvl-qwen3.5-2b-Q4_K_M.gguf}"
MMPROJ="${DVL_MMPROJ:-models/gguf/mmproj-dvl-qwen3.5-2b-F16.gguf}"
LL="${DVL_LLAMA_BIN:-../olmocr-lab/bin/llama-b10909}"
LLAMA_PORT="${DVL_LLAMA_PORT:-8091}"
API_HOST="${DVL_API_HOST:-127.0.0.1}"
API_PORT="${DVL_API_PORT:-8092}"
PY="${DVL_PYTHON:-../iron-coach-th/.venv/bin/python}"
export DVL_LLAMA_URL="http://127.0.0.1:${LLAMA_PORT}"
export DVL_MODEL_NAME="${DVL_MODEL_NAME:-$(basename "$MODEL" .gguf)}"

healthy() { curl -sf -o /dev/null --max-time 2 "$DVL_LLAMA_URL/health"; }

LLAMA_PID=""
cleanup() {
  if [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "[run.sh] stopping llama-server (pid $LLAMA_PID)"
    kill "$LLAMA_PID" 2>/dev/null || true
    for _ in $(seq 20); do kill -0 "$LLAMA_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$LLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if healthy; then
  echo "[run.sh] llama-server already healthy at $DVL_LLAMA_URL — reusing it (will not stop it)"
else
  for f in "$MODEL" "$MMPROJ" "$LL/llama-server"; do [[ -e "$f" ]] || { echo "[run.sh] missing $f" >&2; exit 1; }; done
  [[ "$(basename "$MODEL")" == *Q4_K_M* ]] || echo "[run.sh] WARNING: DVL_ABSTAIN_T default was tuned for Q4_K_M only" >&2
  mkdir -p runs
  LOG="runs/llama-server-$(date +%Y%m%d-%H%M%S).log"
  echo "[run.sh] starting llama-server on :$LLAMA_PORT (log $LOG)"
  LD_LIBRARY_PATH="$LL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$LL/llama-server" -m "$MODEL" --mmproj "$MMPROJ" \
    -ngl 99 -c 4096 -np 1 --jinja --reasoning off --host 127.0.0.1 --port "$LLAMA_PORT" >"$LOG" 2>&1 &
  LLAMA_PID=$!
  for i in $(seq 120); do
    healthy && break
    kill -0 "$LLAMA_PID" 2>/dev/null || { echo "[run.sh] llama-server exited — see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 1
  done
  healthy || { echo "[run.sh] llama-server not healthy after 120 s — see $LOG" >&2; exit 1; }
  echo "[run.sh] llama-server healthy"
fi

echo "[run.sh] API on http://$API_HOST:$API_PORT (model $DVL_MODEL_NAME)"
# ไม่ใช้ exec — ให้ trap ทำงานตอน uvicorn จบ
"$PY" -m uvicorn api.app:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null; wait "$API_PID" 2>/dev/null; exit 130' INT TERM
wait "$API_PID"
