#!/usr/bin/env bash
# Start / stop / status helper for the Reachy Mini local daemon.
#
#   ./scripts/daemon.sh start   # launch daemon in background
#   ./scripts/daemon.sh stop    # stop daemon (robot goes to sleep)
#   ./scripts/daemon.sh status  # show whether it's running
#   ./scripts/daemon.sh logs    # tail the log file
#
# The daemon talks to the Reachy Mini's motor controller over the USB-C
# serial port, and exposes a local HTTP/WebSocket API on localhost:8000
# for the `reachy-mini` Python SDK to connect to.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
PID_FILE="logs/daemon.pid"
LOG_FILE="logs/daemon.log"
OUT_FILE="logs/daemon.stdout.log"
BIN=".venv/bin/reachy-mini-daemon"

case "${1:-}" in
  start)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "daemon already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    [[ -x "$BIN" ]] || { echo "Run 'pip install -r requirements.txt' first"; exit 1; }
    "$BIN" --headless --no-media --localhost-only \
           --log-level INFO --log-file "$LOG_FILE" \
           > "$OUT_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "daemon started (pid $(cat "$PID_FILE"))"
    echo "waiting for motor init ..."
    sleep 6
    tail -n 5 "$OUT_FILE"
    ;;
  stop)
    if [[ -f "$PID_FILE" ]]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "stopping daemon (pid $PID) — robot will go to sleep ..."
        for _ in 1 2 3 4 5 6 7 8; do
          kill -0 "$PID" 2>/dev/null || break
          sleep 0.5
        done
      fi
      rm -f "$PID_FILE"
    else
      echo "daemon not running (no pid file)"
    fi
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "RUNNING (pid $(cat "$PID_FILE"))"
    else
      echo "STOPPED"
    fi
    ;;
  logs)
    tail -n 100 -f "$OUT_FILE"
    ;;
  *)
    echo "usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
