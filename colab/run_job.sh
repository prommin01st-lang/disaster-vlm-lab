#!/usr/bin/env bash
# ใช้: colab/run_job.sh colab/jobs/<job>.sh [GPU=L4] [TIMEOUT_MIN=240]
# exit code = exit code ของ job (124 = timeout/ไม่พบ EXIT_CODE, 125 = ดึงผลไม่ได้, 126 = poll ล้มเหลวติดกัน)
set -euo pipefail
JOB=$1; GPU=${2:-L4}; TIMEOUT_MIN=${3:-240}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME="dvl-$(basename "$JOB" .sh)-$(date +%m%d%H%M%S)"
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"
BUNDLE=$(mktemp --suffix .tar.gz)
TOKFILE=""

cleanup() {
  local rc=$? ok=0 i
  rm -f "$BUNDLE" ${TOKFILE:+"$TOKFILE"}
  for i in 1 2 3; do
    if colab stop -s "$NAME" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" = 1 ]; then echo "[run_job] stopped $NAME"
  else echo "[run_job] !!! STOP FAILED — run: colab stop -s $NAME" >&2; fi
  exit $rc
}
trap cleanup EXIT   # หยุด VM เสมอ แม้ Ctrl+C หรือ error

mkdir -p "$ROOT/scripts"
tar -czf "$BUNDLE" -C "$ROOT" dvl scripts colab requirements-colab.txt
GPU_ARGS=(); [ "$GPU" != "CPU" ] && GPU_ARGS=(--gpu "$GPU")
colab new -s "$NAME" "${GPU_ARGS[@]}"
colab upload -s "$NAME" "$BUNDLE" /content/bundle.tar.gz
if [ -n "${HF_TOKEN:-}" ]; then   # token ผ่านไฟล์ (mode 600) ไม่ผ่าน argv
  TOKFILE=$(mktemp); chmod 600 "$TOKFILE"; printf '%s' "$HF_TOKEN" > "$TOKFILE"
  colab upload -s "$NAME" "$TOKFILE" /content/.hf_token >/dev/null
  rm -f "$TOKFILE"; TOKFILE=""
fi
colab exec -s "$NAME" -f "$ROOT/colab/bootstrap.py" --timeout 900
ENV_ARGS=(--env "JOB=$JOB")
for v in LIMIT TEACHER_MAX TEACHER MODEL ADAPTER TAG MAX_STEPS EPOCHS LR; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(--env "$v=${!v}")
done
colab exec -s "$NAME" -f "$ROOT/colab/launch.py" "${ENV_ARGS[@]}"

set +e   # จากนี้ต้องพยายามดึงผลให้ได้เสมอ
RC=124; FAILS=0
DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
  sleep 60
  if STATUS=$(colab exec -s "$NAME" -f "$ROOT/colab/poll.py" --timeout 60); then FAILS=0
  else STATUS="__POLL_FAIL__"; FAILS=$((FAILS+1))
    [ "$FAILS" -eq 5 ] && echo "[run_job] WARN: 5 poll failures in a row" >&2
    if [ "$FAILS" -ge 15 ]; then echo "[run_job] too many poll failures" >&2; RC=126; break; fi
  fi
  echo "$STATUS" | tail -4
  if echo "$STATUS" | grep -q "__DONE__"; then
    RC=$(echo "$STATUS" | grep -o 'EXIT_CODE=[0-9]*' | tail -1 | cut -d= -f2); RC=${RC:-124}; break
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[run_job] TIMEOUT" >&2; RC=124; break; }
done

FETCHED=0
for i in 1 2 3; do
  if colab exec -s "$NAME" -f "$ROOT/colab/pack_out.py" --timeout 600 \
     && colab download -s "$NAME" /content/out.tar.gz "$OUT/out.tar.gz" \
     && tar -xzf "$OUT/out.tar.gz" -C "$OUT"; then
    rm -f "$OUT/out.tar.gz"; FETCHED=1; break
  fi
  echo "[run_job] fetch attempt $i failed" >&2; sleep 10
done
if [ "$FETCHED" = 0 ]; then
  echo "[run_job] fallback: downloading job.log only" >&2
  colab download -s "$NAME" /content/dvl/out/job.log "$OUT/job.log" || true
  [ "$RC" -eq 0 ] && RC=125
fi
echo "[run_job] results → $OUT (job exit code $RC)"
exit "$RC"
