#!/usr/bin/env bash
set -euo pipefail

HEALTH_URL="http://127.0.0.1:8001/health"

# Fast path: already healthy
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
  echo "[ACE-Step] Server ready."
  exit 0
fi

# Server is managed by systemd (acestep-api.service) — just wait for it to be healthy.
# It may be mid-startup (loading models) or restarting after a crash.
echo "[ACE-Step] Waiting for server to become healthy..."
for i in $(seq 1 300); do
  sleep 2
  if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    echo "[ACE-Step] Server ready (${i}x2s elapsed)."
    exit 0
  fi
done

echo "[ACE-Step] Timeout after 600s. Check: journalctl --user -u acestep-api -n 50" >&2
exit 1
