#!/usr/bin/env bash
# ใช้: colab/run_job.sh colab/jobs/<job>.sh [GPU=L4] [TIMEOUT_MIN=240]
set -euo pipefail
JOB=$1; GPU=${2:-L4}; TIMEOUT_MIN=${3:-240}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME="dvl-$(basename "$JOB" .sh)-$(date +%m%d%H%M)"
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"
BUNDLE=$(mktemp --suffix .tar.gz)

cleanup() { colab stop -s "$NAME" >/dev/null 2>&1 || true; rm -f "$BUNDLE"; echo "[run_job] stopped $NAME"; }
trap cleanup EXIT   # หยุด VM เสมอ แม้ Ctrl+C หรือ error

mkdir -p "$ROOT/scripts"
tar -czf "$BUNDLE" -C "$ROOT" dvl scripts colab requirements-colab.txt
GPU_ARGS=(); [ "$GPU" != "CPU" ] && GPU_ARGS=(--gpu "$GPU")
colab new -s "$NAME" "${GPU_ARGS[@]}"
colab upload -s "$NAME" "$BUNDLE" /content/bundle.tar.gz
colab exec -s "$NAME" -f "$ROOT/colab/bootstrap.py" --timeout 900
ENV_ARGS=(--env "JOB=$JOB")
for v in HF_TOKEN LIMIT TEACHER MODEL ADAPTER TAG MAX_STEPS EPOCHS LR; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(--env "$v=${!v}")
done
colab exec -s "$NAME" -f "$ROOT/colab/launch.py" "${ENV_ARGS[@]}"

DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
  sleep 60
  STATUS=$(colab exec -s "$NAME" -f "$ROOT/colab/poll.py" --timeout 60 || echo "__POLL_FAIL__")
  echo "$STATUS" | tail -4
  echo "$STATUS" | grep -q "__DONE__" && break
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[run_job] TIMEOUT"; break; }
done

colab exec -s "$NAME" -f "$ROOT/colab/pack_out.py" --timeout 600
colab download -s "$NAME" /content/out.tar.gz "$OUT/out.tar.gz"
tar -xzf "$OUT/out.tar.gz" -C "$OUT" && rm "$OUT/out.tar.gz"
echo "[run_job] results → $OUT"
grep EXIT_CODE "$OUT/out/job.log"
