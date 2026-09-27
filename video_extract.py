#!/usr/bin/env python3
"""
video_extract.py — Estrazione audio→transcript + keyframe da video.

Pipeline:
  1. Estrae audio (ffmpeg → WAV 16kHz mono)
  2. Trascrive (faster-whisper)
  3. Scene detection (PySceneDetect) + estrazione keyframe
  4. Salva transcript.md, keyframes/, video_meta.json

Usage:
  python video_extract.py --input video.mp4 --output ./output \\
      [--whisper-model base] [--scene-threshold 0.4] [--max-frames 50]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from faster_whisper import WhisperModel


def _preload_nvidia_cuda_libs() -> None:
    """dlopen NVIDIA pip wheels by absolute path (ctypes, RTLD_GLOBAL) so
    ctranslate2 finds libcublas/cudnn without system CUDA toolkit or
    LD_LIBRARY_PATH. No-op on CPU-only machines.
    # ponytail: loads every .so in nvidia/*/lib; per-lib selection if startup cost matters."""
    import ctypes
    import glob as _glob
    from pathlib import Path as _Path
    base = _Path(sys.prefix)
    libs = sorted(_glob.glob(str(base / "lib" / "python*" / "site-packages"
                                    / "nvidia" / "*" / "lib" / "*.so*")))
    for _pass in (1, 2):  # 2nd pass retries libs whose deps load later
        for so in libs:
            if _Path(so).is_dir():
                continue
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_nvidia_cuda_libs()

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Return True iff ffmpeg is on PATH and works."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def format_timestamp(seconds: float) -> str:
    """Seconds → [HH:MM:SS]."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


SUPPORTED_FORMATS = (".mp4", ".mov", ".avi", ".mkv")


# ---------------------------------------------------------------------------
# Phase 1 – Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, output_wav: str) -> None:
    """Extract PCM 16-bit 16 kHz mono audio to *output_wav* via ffmpeg."""
    print(f"[audio] Extracting audio from {video_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.strip()}")
    print(f"[audio] Audio extracted → {output_wav}")


# ---------------------------------------------------------------------------
# Phase 2 – Transcription
# ---------------------------------------------------------------------------

def _run_transcribe(model, audio_path: str) -> list[tuple[float, str]]:
    """Run model.transcribe + materialize segments (errors surface here,
    inside the caller's try so auto-fallback catches transcribe-time CUDA
    failures, not just model-load ones)."""
    print(f"[transcribe] Transcribing ...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    print(f"[transcribe] Language: {info.language} "
          f"(p={info.language_probability:.2f})")

    result: list[tuple[float, str]] = []
    for seg in segments:
        result.append((seg.start, seg.text.strip()))
    return result


def transcribe(audio_path: str, model_name: str = "base",
               device: str = "auto") -> list[tuple[float, str]]:
    """Transcribe *audio_path* via faster-whisper.

    ``device``: ``auto`` (try CUDA, fallback CPU), ``cuda``, ``cpu``.
    Returns list of ``(start_seconds, text)``.
    """
    print(f"[transcribe] Loading whisper model '{model_name}' "
          f"(device={device}) ...")

    if device == "cpu":
        dev, compute_type = "cpu", "int8"
    elif device == "cuda":
        dev, compute_type = "cuda", "int8"
    else:  # auto
        dev, compute_type = "cuda", "int8"

    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=dev,
            compute_type=compute_type,
        )
        return _run_transcribe(model, audio_path)
    except Exception as exc:
        if device == "auto":
            print(f"[transcribe] CUDA failed ({exc}), falling back to CPU ...")
            try:
                model = WhisperModel(
                    model_size_or_path=model_name,
                    device="cpu",
                    compute_type="int8",
                )
                return _run_transcribe(model, audio_path)
            except Exception as exc2:
                raise RuntimeError(
                    f"Failed to load whisper model '{model_name}': {exc2}"
                ) from exc2
        else:
            raise RuntimeError(
                f"Failed to load whisper model '{model_name}': {exc}"
            ) from exc


def save_transcript(data: list[tuple[float, str]], dest: Path) -> None:
    """Write transcript.md with lines ``[HH:MM:SS] text``."""
    print(f"[transcript] Saving → {dest}")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("# Transcript\n\n")
        for start_sec, text in data:
            f.write(f"{format_timestamp(start_sec)} {text}\n")


# ---------------------------------------------------------------------------
# Phase 3 – Scene detection & keyframes
# ---------------------------------------------------------------------------

def detect_scenes_and_extract_frames(
    video_path: str,
    output_dir: str,
    threshold: float = 0.4,
    max_frames: int = 50,
) -> int:
    """Detect scene changes & save keyframe JPGs.  Returns count saved."""
    print(f"[scenes] Detecting scene changes (threshold={threshold}) ...")

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()  # list of (FrameTimecode, FrameTimecode)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)

    # Decide which timestamps (seconds) get a frame
    timestamps: list[float] = [0.0]  # always include first frame

    for sc in scene_list:
        timestamps.append(float(sc[0]))  # sc[0] = scene start FrameTimecode

    # Deduplicate near-zero differences
    cleaned: list[float] = []
    for ts in timestamps:
        if not cleaned or abs(ts - cleaned[-1]) > 0.04:
            cleaned.append(ts)
    timestamps = cleaned

    # Apply --max-frames cap by uniform downsampling
    if max_frames and len(timestamps) > max_frames:
        print(f"[scenes] Max frames limit ({max_frames}) hit: "
              f"downsampling {len(timestamps)} → {max_frames}")
        step = len(timestamps) / max_frames
        timestamps = [timestamps[min(int(i * step), len(timestamps) - 1)]
                      for i in range(max_frames)]

    saved = 0
    for ts in timestamps:
        frame_no = min(int(ts * fps), frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_no, 0))
        ret, frame = cap.read()
        if ret:
            out = os.path.join(output_dir, f"frame_{saved + 1:03d}.jpg")
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

    cap.release()
    print(f"[scenes] Saved {saved} keyframes → {output_dir}")
    return saved


# ---------------------------------------------------------------------------
# Phase 4 – Metadata
# ---------------------------------------------------------------------------

def get_video_metadata(video_path: Path) -> dict:
    """Read basic video metadata via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "file": video_path.name,
        "duration_seconds": round(duration, 2),
        "duration_formatted": format_timestamp(duration),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract transcript + keyframes from video.",
    )
    p.add_argument("--input", "-i", required=True, help="Input video file")
    p.add_argument(
        "--output", "-o", required=True,
        help="Output directory (created if missing)",
    )
    p.add_argument(
        "--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base)",
    )
    p.add_argument(
        "--scene-threshold", type=float, default=0.4,
        help="Scene-detection sensitivity (default: 0.4, lower = more cuts)",
    )
    p.add_argument(
        "--max-frames", type=int, default=50,
        help="Max keyframes to extract (default: 50, 0 = unlimited)",
    )
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for whisper inference: auto (try CUDA, fallback CPU), "
             "cuda, cpu (default: auto)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    output_dir = Path(args.output)

    # ---- Validation ----
    if not video_path.exists():
        print(f"[error] Input not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    ext = video_path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        print(f"[error] Unsupported format '{ext}'. "
              f"Supported: {', '.join(SUPPORTED_FORMATS)}",
              file=sys.stderr)
        sys.exit(1)

    if not check_ffmpeg():
        print("[error] ffmpeg not found.  Install it, e.g.:\n"
              "  apt install ffmpeg          # Debian/Ubuntu\n"
              "  brew install ffmpeg          # macOS\n"
              "  pacman -S ffmpeg             # Arch",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Pipeline ----
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_wav = os.path.join(tmpdir, "audio.wav")

        # Phase 1 — audio
        print("\n=== Phase 1: Audio extraction ===")
        extract_audio(video_path, temp_wav)

        # Phase 2 — transcription
        print("\n=== Phase 2: Transcription ===")
        transcript_data = transcribe(temp_wav, args.whisper_model, args.device)
        save_transcript(transcript_data, output_dir / "transcript.md")
        print(f"[transcript] {len(transcript_data)} segment(s)")

        # Phase 3 — scene detection & keyframes
        print("\n=== Phase 3: Scene detection & keyframes ===")
        keyframes_dir = output_dir / "keyframes"
        limit = args.max_frames if args.max_frames > 0 else 1_000_000
        num_frames = detect_scenes_and_extract_frames(
            str(video_path),
            str(keyframes_dir),
            threshold=args.scene_threshold,
            max_frames=limit,
        )

        # Phase 4 — metadata
        print("\n=== Phase 4: Metadata ===")
        meta = get_video_metadata(video_path)
        meta["num_transcript_segments"] = len(transcript_data)
        meta["num_keyframes"] = num_frames
        meta_path = output_dir / "video_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[meta] Saved → {meta_path}")
        print(f"[meta] Duration: {meta['duration_formatted']}, "
              f"Frames: {meta['total_frames']}, FPS: {meta['fps']}, "
              f"Size: {meta['width']}×{meta['height']}")

    # tmpdir auto-cleaned
    print("\n=== Done ===")
    print(f"  {output_dir.resolve()}/")
    print(f"  ├── transcript.md      ({len(transcript_data)} segments)")
    print(f"  ├── keyframes/          ({num_frames} images)")
    print(f"  └── video_meta.json    (metadata)")


if __name__ == "__main__":
    main()
