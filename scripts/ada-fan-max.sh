#!/usr/bin/env bash
set -euo pipefail

PINCTRL="$(command -v pinctrl || true)"
if [[ -z "$PINCTRL" ]]; then
  echo "pinctrl is unavailable; cannot control FAN_PWM." >&2
  exit 1
fi

while true; do
  # Pironman 5 Pro Max fan PWM is active-low. Driving the named pin low
  # continuously is the hardware-confirmed full-speed setting.
  "$PINCTRL" FAN_PWM op dl
  sleep 5
done
