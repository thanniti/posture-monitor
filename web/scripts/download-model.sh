#!/usr/bin/env bash
# web/scripts/download-model.sh
set -euo pipefail
mkdir -p "$(dirname "$0")/../public/models"
curl -sSfL -o "$(dirname "$0")/../public/models/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Downloaded public/models/pose_landmarker_lite.task"
