#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install -D -m 0755 "$PROJECT_DIR/scripts/ada-fan-max.sh" /usr/local/lib/ada-pi/ada-fan-max.sh
install -D -m 0644 "$PROJECT_DIR/scripts/ada-fan-max.service" /etc/systemd/system/ada-fan-max.service
systemctl daemon-reload
systemctl enable --now ada-fan-max.service
echo "ADA maximum-fan service installed and running."
