#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$0")/../assets"
curl -sL -o "$(dirname "$0")/../assets/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Downloaded assets/pose_landmarker_lite.task"
