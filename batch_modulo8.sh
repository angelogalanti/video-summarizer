#!/usr/bin/env bash
# Batch estrazione video 02-08 Morpheus Modulo 8
# Esegue in sequenza per non sovraccaricare la GPU (GTX 1080 8GB)
set -euo pipefail

SRC="/mnt/c/Users/x/External HD/Investing/Morpheus/Order Flow/Parte 2/Modulo 8 Orderflow Intermedio e Strategia Nasdaq"
BASE_OUT="$HOME/content/video-analysis/Morpheus-OrderFlow-Modulo8"
SCRIPT="$HOME/repos/video-summarizer/video_extract.py"
PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python3"

declare -A MAP
MAP[02]="2. Esempi Pratici_1080p.mp4 → 02-Esempi-Pratici"
MAP[03]="3. Footprint e Fixed Delta Profile_ Esempi Pratici_1080p.mp4 → 03-Footprint-Fixed-Delta-Profile"
MAP[04]="4. Q&A_1080p.mp4 → 04-Q&A"
MAP[05]="5. Dinamiche Orderflow per Scalp su NQ_1080p.mp4 → 05-Dinamiche-Orderflow-Scalp-NQ"
MAP[06]="6. Q&A n.2_1080p.mp4 → 06-Q&A-n2"
MAP[07]="7. Profile e VWAP come Bias Medium Term_1080p.mp4 → 07-Profile-VWAP-Bias-Medium-Term"
MAP[08]="8. Deviazioni standard e Commentary_1080p.mp4 → 08-Deviazioni-standard-e-Commentary"

for N in 02 03 04 05 06 07 08; do
    ENTRY="${MAP[$N]}"
    FILE="${ENTRY%% → *}"
    DIR="${ENTRY##* → }"

    INPUT="$SRC/$FILE"
    OUTPUT="$BASE_OUT/$DIR"

    echo ""
    echo "============================================================"
    echo "  Video $N — $FILE"
    echo "  Output → $OUTPUT"
    echo "============================================================"
    echo ""

    $PYTHON "$SCRIPT" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --whisper-model small \
        --max-frames 15 \
        --device auto

    echo ""
    echo "  ✓ Video $N completato"
done

echo ""
echo "============================================================"
echo "  BATCH COMPLETATO — tutti i 7 video (02-08)"
echo "============================================================"
