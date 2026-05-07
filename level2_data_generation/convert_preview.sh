#!/bin/bash
# Convert all pred.mp4 to pred_preview.mp4 for cursor viewing
MODELS="happyhorse wanx21_i2v_plus"
BASE="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp"

for model in $MODELS; do
  find "$BASE/$model" -name "pred.mp4" | while read src; do
    out="${src%pred.mp4}pred_preview.mp4"
    if [ -f "$out" ]; then
      echo "[skip] $out"
      continue
    fi
    echo "[conv] $src"
    ffmpeg -y -i "$src" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -crf 18 \
      "$out" -loglevel error
    echo "[done] $out"
  done
done
echo "ALL CONVERTED"
