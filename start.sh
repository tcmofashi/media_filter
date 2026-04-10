#!/usr/bin/env bash
# XPfilter Linux 唯一入口

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BACKEND_PORT=31211
FRONTEND_PORT=31212
MODE=all
BUILD_WEBUI=false
CLEAN_CACHE=false

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

usage() {
cat <<'EOF'
Usage:
  ./start.sh [mode] [options]

Modes:
  all        Start backend API + webui (default)
  api        Start backend API only
  frontend   Start webui only (backend not started)

Options:
  --del-cache       Clear cache directories before start
  --build-webui     Build frontend once before starting
  --help            Show this help

Notes:
  Linux 下当前仓库统一入口为 start.sh，建议通过该脚本启动服务与调度面板。
EOF
}

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing dependency: $name"
    echo "请先安装 $name 并确保在 PATH 中可用"
    exit 1
  fi
}

kill_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" | xargs -r kill -9 2>/dev/null || true
  fi
}

run_api() {
  echo "[api] starting on 0.0.0.0:${BACKEND_PORT}"
  cd "$PROJECT_ROOT"
  python src/main.py >"$BACKEND_LOG" 2>&1 &
  API_PID=$!
}

run_frontend() {
  echo "[webui] starting on 0.0.0.0:${FRONTEND_PORT}"
  cd "$PROJECT_ROOT/webui"
  if [[ "$BUILD_WEBUI" == true ]]; then
    npm run build >"$LOG_DIR/frontend_build.log" 2>&1
    npm run preview -- --host 0.0.0.0 --port "$FRONTEND_PORT" >"$FRONTEND_LOG" 2>&1 &
  else
    npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" >"$FRONTEND_LOG" 2>&1 &
  fi
  FRONTEND_PID=$!
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    all|api|frontend)
      MODE="$1"
      shift
      ;;
    --del-cache)
      CLEAN_CACHE=true
      shift
      ;;
    --build-webui)
      BUILD_WEBUI=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$CLEAN_CACHE" == true ]]; then
  echo "[cache] clearing optional cache dirs"
  rm -rf "$PROJECT_ROOT/data/cache/thumbnails"/* 2>/dev/null || true
  rm -rf "$PROJECT_ROOT/data/cache/screenshots"/* 2>/dev/null || true
  rm -rf "$PROJECT_ROOT/data/cache/transcodes"/* 2>/dev/null || true
fi

if [[ "$MODE" == "all" || "$MODE" == "api" ]]; then
  require_cmd python
fi

if [[ "$MODE" == "all" || "$MODE" == "frontend" ]]; then
  require_cmd npm
fi

TOTAL_STEPS=1
[[ "$MODE" == "all" ]] && TOTAL_STEPS=2
STEP=0
echo "=== XPfilter 启动器 (mode=${MODE}) ==="

if [[ "$MODE" == "all" || "$MODE" == "api" ]]; then
  STEP=$((STEP + 1))
  echo "[${STEP}/${TOTAL_STEPS}] 释放后端端口 ${BACKEND_PORT} ..."
  kill_port "$BACKEND_PORT"
  run_api
fi

if [[ "$MODE" == "all" || "$MODE" == "frontend" ]]; then
  if [[ "$MODE" == "all" ]]; then
    STEP=$((STEP + 1))
  else
    STEP=1
  fi
  echo "[${STEP}/${TOTAL_STEPS}] 释放前端端口 ${FRONTEND_PORT} ..."
  kill_port "$FRONTEND_PORT"
  run_frontend
fi

echo "=== 启动完成 ==="
if [[ "$MODE" == "all" || "$MODE" == "api" ]]; then
  echo "后端:  http://localhost:${BACKEND_PORT}"
  echo "API文档: http://localhost:${BACKEND_PORT}/docs"
fi
if [[ "$MODE" == "all" || "$MODE" == "frontend" ]]; then
  echo "前端:  http://localhost:${FRONTEND_PORT}"
  echo "前端日志: ${FRONTEND_LOG}"
fi
if [[ "$MODE" == "all" ]]; then
  echo "后端日志: ${BACKEND_LOG}"
fi

echo "按 Ctrl+C 退出"
if [[ "$MODE" == "all" ]]; then
  trap "kill ${API_PID:-} ${FRONTEND_PID:-} 2>/dev/null || true" EXIT
  wait
elif [[ "$MODE" == "api" ]]; then
  trap "kill ${API_PID:-} 2>/dev/null || true" EXIT
  wait
else
  trap "kill ${FRONTEND_PID:-} 2>/dev/null || true" EXIT
  wait
fi
