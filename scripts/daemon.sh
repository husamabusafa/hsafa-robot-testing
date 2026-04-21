#!/usr/bin/env bash
# Start / stop / status helper for the Reachy Mini local daemon.
#
#   ./scripts/daemon.sh start    # launch daemon in background
#   ./scripts/daemon.sh stop     # stop daemon (robot goes to sleep)
#   ./scripts/daemon.sh restart  # stop + start
#   ./scripts/daemon.sh status   # show whether it's running
#   ./scripts/daemon.sh logs     # tail the log file
#
# The daemon talks to the Reachy Mini's motor controller over the USB-C
# serial port, and exposes a local HTTP/WebSocket API on localhost:8000
# for the `reachy-mini` Python SDK to connect to.
#
# IMPORTANT: the daemon is started WITHOUT ``--no-media`` so it owns the
# camera + audio hardware. ``main.py`` then pulls frames and routes Gemini
# audio through ``reachy.media`` (GStreamer pipeline). If you need raw
# OpenCV / sounddevice access instead, re-add ``--no-media`` below OR call
# ``mini.release_media()`` from Python (idempotent, see SDK docs).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
PID_FILE="logs/daemon.pid"
LOG_FILE="logs/daemon.log"
OUT_FILE="logs/daemon.stdout.log"
BIN=".venv/bin/reachy-mini-daemon"

start_daemon() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "daemon already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  [[ -x "$BIN" ]] || { echo "Run 'pip install -r requirements.txt' first"; exit 1; }
  "$BIN" --headless --localhost-only \
         --log-level INFO --log-file "$LOG_FILE" \
         > "$OUT_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  PID=$!
  echo "daemon started (pid $PID)"
  echo "waiting for motor + media init (port 8000) ..."

  # GStreamer webrtcsink codec discovery can take 15+ s on macOS, so
  # actively poll instead of sleeping a fixed amount.
  for i in $(seq 1 60); do
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "ERROR: daemon process died during startup. Last log lines:"
      tail -n 30 "$OUT_FILE"
      rm -f "$PID_FILE"
      exit 1
    fi
    if nc -z localhost 8000 2>/dev/null; then
      echo "daemon ready after ${i}s"
      return 0
    fi
    sleep 1
  done

  echo "ERROR: daemon did not bind to port 8000 within 60s. Last log lines:"
  tail -n 30 "$OUT_FILE"
  exit 1
}

stop_daemon() {
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
}

case "${1:-}" in
  start)   start_daemon ;;
  stop)    stop_daemon ;;
  restart) stop_daemon; sleep 1; start_daemon ;;
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
    echo "usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
