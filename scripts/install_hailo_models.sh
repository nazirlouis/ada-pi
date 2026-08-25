#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ADA_HAILO_MODEL_DIR:-$PROJECT_DIR/data/models/hailo8}"
# HailoRT 4.23/TAPPAS 5.1 on Raspberry Pi targets Model Zoo v2.17.
BASE_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8"

mkdir -p "$MODEL_DIR"

download_model() {
  local name="$1" expected="$2" temporary
  temporary="$(mktemp "$MODEL_DIR/.${name}.XXXXXX")"
  trap 'rm -f "$temporary"' RETURN
  curl -fL --retry 3 "$BASE_URL/$name" -o "$temporary"
  echo "$expected  $temporary" | sha256sum --check --status
  mv "$temporary" "$MODEL_DIR/$name"
  trap - RETURN
}

download_model yolov8m.hef 9481dbff7798d90302e170958943578d444b61c9833c67fef36075fe129efe7f
download_model yolov8m_pose.hef f39b0fb38f3e57c91af885f35145684b734955b79d48edaab1cffcdcd6bfc5a1

echo "Installed Hailo-8 models in $MODEL_DIR"
