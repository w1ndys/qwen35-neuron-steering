#!/bin/bash
# 一键入口：./run.sh chat | ./run.sh exp | ./run.sh calib | ./run.sh smoke
set -euo pipefail
cd "$(dirname "$0")"

PY="$HOME/.venvs/qwen-steer/bin/python"
if [ ! -x "$PY" ]; then
  echo "找不到 venv: $PY" >&2
  echo "先跑:" >&2
  echo "  python3 -m venv ~/.venvs/qwen-steer" >&2
  echo "  ~/.venvs/qwen-steer/bin/pip install -U torch transformers accelerate safetensors" >&2
  exit 1
fi

# 让慢 fallback 的警告安静一点（不影响正确性）
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

cmd="${1:-chat}"
shift || true

case "$cmd" in
  chat)  exec "$PY" chat.py "$@" ;;
  exp)   exec "$PY" experiment.py "$@" ;;
  calib) exec "$PY" calibrate.py "$@" ;;
  smoke) exec "$PY" smoke.py "$@" ;;
  selftest) exec "$PY" selftest.py "$@" ;;
  *)     echo "用法: $0 {chat|exp|calib|smoke|selftest} [args...]" >&2; exit 2 ;;
esac
