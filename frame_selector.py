#!/usr/bin/env python3
"""
frame_selector.py — Raggruppa frame simili via percettive hash e seleziona
solo quelli distinti.

Algoritmo:
  1. Carica tutti i frame .jpg in input (ordinati per nome)
  2. Calcola perceptual hash (phash) per ogni frame
  3. Raggruppa frame consecutivi con hash distance < threshold
  4. Prende il primo frame di ogni gruppo come rappresentativo
  5. Copia i frame rappresentativi nella directory output
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import imagehash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raggruppa frame simili via perceptual hash e seleziona i frame distinti."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory contenente i frame .jpg",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory destinazione per i frame selezionati",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Soglia phash distance (default: 10, più alto = più aggregazione)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stampa dettagli su gruppi e frame scartati",
    )
    return parser.parse_args()


def load_frames(input_dir: str) -> list[Path]:
    """Ritorna lista di path .jpg ordinati per nome."""
    frames = sorted(Path(input_dir).glob("*.jpg"))
    if not frames:
        print(f"ERRORE: Nessun file .jpg trovato in {input_dir}", file=sys.stderr)
        sys.exit(1)
    return frames


def compute_phash(frame_path: Path) -> imagehash.ImageHash | None:
    """Calcola phash di un frame. Ritorna None se l'immagine è corrotta."""
    try:
        with Image.open(frame_path) as img:
            # Forza RGB per consistenza
            img = img.convert("RGB")
            return imagehash.phash(img)
    except (UnidentifiedImageError, OSError, Exception) as e:
        print(f"  AVVISO: impossibile leggere {frame_path.name}: {e}", file=sys.stderr)
        return None


def group_frames(
    frames: list[Path], threshold: int, verbose: bool
) -> list[list[Path]]:
    """
    Raggruppa frame consecutivi in base alla distanza phash.
    Ogni gruppo contiene frame consecutivi simili tra loro.
    """
    if not frames:
        return []

    groups: list[list[Path]] = []
    current_group: list[Path] = []
    prev_hash = None

    for frame in frames:
        h = compute_phash(frame)
        if h is None:
            continue  # salta frame corrotti, non li mette in nessun gruppo

        if prev_hash is not None:
            distance = h - prev_hash
            if distance >= threshold:
                # Distanza troppo alta → nuovo gruppo
                if current_group:
                    groups.append(current_group)
                current_group = [frame]
            else:
                # Frame simile al precedente → stesso gruppo
                current_group.append(frame)
        else:
            # Primo frame valido
            current_group = [frame]

        prev_hash = h

    # Aggiunge l'ultimo gruppo
    if current_group:
        groups.append(current_group)

    if verbose:
        total_input = sum(len(g) for g in groups)
        total_selected = len(groups)
        print(f"\n  Frame validi elaborati: {total_input}")
        print(f"  Gruppi trovati:         {total_selected}")
        print(f"  Frame scartati:         {total_input - total_selected}")
        print()

    return groups


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    threshold = args.threshold
    verbose = args.verbose

    # Verifica input
    if not os.path.isdir(input_dir):
        print(f"ERRORE: input directory non trovata: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Crea output
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Soglia: {threshold}")

    # Carica frame
    frames = load_frames(input_dir)
    if verbose:
        print(f"Trovati {len(frames)} file .jpg")

    # Raggruppa
    groups = group_frames(frames, threshold, verbose)

    if verbose:
        print(f"Gruppi formati: {len(groups)}")

    # Copia il primo frame di ogni gruppo
    selected_count = 0
    for i, group in enumerate(groups, 1):
        rep = group[0]
        dst = Path(output_dir) / rep.name
        shutil.copy2(rep, dst)
        selected_count += 1

        if verbose:
            discarded = len(group) - 1
            discarded_info = f" (+{discarded} scartati)" if discarded else ""
            print(f"  [{i}/{len(groups)}] {rep.name}{discarded_info}")

    print(f"\nFatto. Copiati {selected_count} frame distinti su {len(frames)} totali.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
