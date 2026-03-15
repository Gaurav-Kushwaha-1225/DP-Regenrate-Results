#!/usr/bin/env python3
"""
compute_partC_descriptors.py

Part-C descriptor pipeline (QMPSA / polar-nonpolar surface areas) for the Nature Chemistry 2021
paper (s41557-021-00646-w).

Goal (Part C):
- Populate uppercase QMPSA-like descriptors:
    QPSA, QNPSA
  (and optionally overwrite Overall surface area if you want it to match the chosen "Q-surface")

Why a separate script from Part-B?
- Part-B script ingests a Multiwfn log and (by design) stores the LAST "Summary of surface analysis"
  per Multiwfn session. If you run multiple surface definitions in one session (e.g., Electron density
  iso=0.001 and Function 25 iso=0.001), you need BOTH blocks.
- Part-C therefore extracts EVERY surface-analysis block and stores them as separate JSON records
  labeled by surface definition, then uses the selected surface-definition tag to compute QPSA/QNPSA.

What this script does
1) Ingest Multiwfn terminal logs (module 12) and write structured JSON files next to each fragment
   .wfx file:
      <basename>.surface.<tag>.json
   Example tag:
      edens_iso0.001
      func25_iso0.001

2) Update my_dataset.csv:
   - QPSA = polar_area(sugar) + polar_area(interactor)   [on the selected Q-surface tag]
   - QNPSA = nonpolar_area(sugar) + nonpolar_area(interactor)

Notes
- "polar_area" and "nonpolar_area" are read from Multiwfn summary lines:
     Polar surface area (|ESP| > 10 kcal/mol): ...
     Nonpolar surface area (|ESP| <= 10 kcal/mol): ...
- Default polar threshold is 10 kcal/mol in Multiwfn; if you changed it in Multiwfn advanced options,
  it will be parsed and stored when detectable.

Usage
-----
# 1) ingest raw logs
python compute_partC_descriptors.py --ingest Multiwfn-Results.txt

# 2) update dataset using a chosen surface tag (default: func25_iso0.001)
python compute_partC_descriptors.py --update --q-tag func25_iso0.001

# optional: overwrite existing values
python compute_partC_descriptors.py --update --q-tag func25_iso0.001 --overwrite

Folder layout expected (same style as compute_homo_lumo.py):
DP-Regenrate-Results/
  my_dataset.csv
  calculations/
    GlcNAc/GlcNAc-2/GlcNAc_InterFrag.wfx
    Ala/Ala-2/Ala_InterFrag.wfx
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CALCULATIONS_DIR = BASE_DIR / "calculations"
MY_DATASET_CSV = BASE_DIR / "my_dataset.csv"


# -------------------------
# Parsing utilities
# -------------------------

@dataclass
class SurfaceBlock:
    surface_def_raw: str
    iso: Optional[float]

    overall_surface_A2: Optional[float]
    polar_area_A2: Optional[float]
    nonpolar_area_A2: Optional[float]
    polar_threshold_kcal: Optional[float]

    # Extra (nice to keep, but not required for QPSA/QNPSA)
    volume_A3: Optional[float] = None

def _f(pat: str, s: str) -> Optional[float]:
    m = re.search(pat, s, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def _make_tag(surface_def_raw: str, iso: Optional[float]) -> str:
    """
    Convert Multiwfn's 'current: ...' surface-def string into a stable file tag.
    """
    t = surface_def_raw.strip().lower()
    # common
    t = re.sub(r"\s+", "_", t)
    t = t.replace("(", "").replace(")", "").replace(",", "")
    # shorten
    t = t.replace("electron_density", "edens")
    t = t.replace("function__", "func")
    t = t.replace("function_", "func")
    # only keep safe chars
    t = re.sub(r"[^a-z0-9_+-]", "", t)
    if iso is not None:
        return f"{t}_iso{iso:.3f}"
    return t

def parse_multiwfn_surface_blocks(text: str) -> List[Tuple[str, SurfaceBlock]]:
    """
    Parse ALL Multiwfn surface-analysis blocks from a log.

    Returns:
      list of (wfx_path, SurfaceBlock)
    """
    # 1) find all sessions (each 'Loaded ... wfx successfully')
    load_re = re.compile(r"Loaded\s+(.+?\.wfx)\s+successfully!", re.MULTILINE)
    loads = list(load_re.finditer(text))
    if not loads:
        return []

    out: List[Tuple[str, SurfaceBlock]] = []

    # Helper: slice session text
    for i, m in enumerate(loads):
        wfx_path = m.group(1).strip()
        start = m.end()
        end = loads[i + 1].start() if i + 1 < len(loads) else len(text)
        session = text[start:end]

        # surface definition lines (printed in module 12 menu)
        # Example:
        #   1 Select the way to define surface, current: Electron density, iso:  0.00100
        #   1 Select the way to define surface, current: Function  25, iso:  0.00100
        sdef_re = re.compile(
            r"Select the way to define surface,\s*current:\s*(.+?),\s*iso:\s*([0-9.]+)",
            re.MULTILINE,
        )

        # For each summary block, find the closest PRECEDING surface-def line.
        sum_re = re.compile(r"=+\s*Summary of surface analysis\s*=+", re.MULTILINE)
        sums = list(sum_re.finditer(session))
        if not sums:
            continue

        sdefs = list(sdef_re.finditer(session))
        # If surface-def line is not found, we still try but tag becomes "unknown"
        for sm in sums:
            # find closest sdef before sm
            chosen_raw = "unknown"
            chosen_iso: Optional[float] = None
            for sd in sdefs:
                if sd.start() < sm.start():
                    chosen_raw = sd.group(1).strip()
                    chosen_iso = float(sd.group(2))
                else:
                    break

            # extract the block text: from summary header to "Surface analysis finished!"
            tail = session[sm.start():]
            fin = tail.find("Surface analysis finished!")
            block = tail[:fin] if fin != -1 else tail

            overall_A2 = _f(r"Overall surface area:.*\(\s*([0-9.]+)\s*Angstrom\^2\)", block)
            vol_A3 = _f(r"Volume:.*\(\s*([0-9.]+)\s*Angstrom\^3\)", block)

            nonpolar_A2 = _f(r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\):\s*([0-9.]+)\s*Angstrom\^2", block)
            polar_A2 = _f(r"Polar surface area\s*\(\|ESP\|\s*>\s*([0-9.]+)\s*kcal/mol\):\s*([0-9.]+)\s*Angstrom\^2", block)

            # capture threshold if present in either line
            thr1 = _f(r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\)", block)
            thr2 = _f(r"Polar surface area\s*\(\|ESP\|\s*>\s*([0-9.]+)\s*kcal/mol\)", block)
            thr = thr1 if thr1 is not None else thr2

            # In regex above, group(1) is threshold and group(2) is area; _f returns group(1),
            # so re-parse areas with proper capture:
            mnp = re.search(r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\):\s*([0-9.]+)\s*Angstrom\^2", block)
            mpl = re.search(r"Polar surface area\s*\(\|ESP\|\s*>\s*([0-9.]+)\s*kcal/mol\):\s*([0-9.]+)\s*Angstrom\^2", block)
            if mnp:
                thr = float(mnp.group(1))
                nonpolar_A2 = float(mnp.group(2))
            if mpl:
                thr = float(mpl.group(1))
                polar_A2 = float(mpl.group(2))

            sb = SurfaceBlock(
                surface_def_raw=chosen_raw,
                iso=chosen_iso,
                overall_surface_A2=overall_A2,
                polar_area_A2=polar_A2,
                nonpolar_area_A2=nonpolar_A2,
                polar_threshold_kcal=thr,
                volume_A3=vol_A3,
            )
            out.append((wfx_path, sb))

    return out


# -------------------------
# File discovery / writing
# -------------------------

def _resolve_wfx_to_local_path(wfx_path: str) -> Optional[Path]:
    """
    Convert an absolute or relative path printed in Multiwfn log to an existing local path.
    We try:
      - as-is
      - if it contains 'DP-Regenrate-Results', take the suffix after it and join BASE_DIR
    """
    p = Path(wfx_path.strip())
    if p.exists():
        return p

    # Normalize "Desktop/DP-Regenrate-Results/..." cases
    s = wfx_path.strip()
    key = "DP-Regenrate-Results"
    if key in s:
        suffix = s.split(key, 1)[1].lstrip("/\\")
        cand = BASE_DIR / suffix
        if cand.exists():
            return cand

    # Try join if relative
    cand = BASE_DIR / s
    if cand.exists():
        return cand

    return None

def write_surface_json(wfx_path: Path, sb: SurfaceBlock) -> Path:
    tag = _make_tag(sb.surface_def_raw, sb.iso)
    out_path = wfx_path.parent / f"{wfx_path.stem}.surface.{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(sb) | {"tag": tag, "wfx": str(wfx_path)}, f, indent=2)
    return out_path

def ingest_logs(log_paths: List[Path]) -> None:
    blobs: List[str] = []
    for lp in log_paths:
        blobs.append(lp.read_text(encoding="utf-8", errors="ignore"))
    text = "\n\n".join(blobs)

    blocks = parse_multiwfn_surface_blocks(text)
    if not blocks:
        print("No Multiwfn surface-analysis blocks found.")
        return

    written = 0
    for wfx_str, sb in blocks:
        wfx_local = _resolve_wfx_to_local_path(wfx_str)
        if not wfx_local:
            print(f"[WARN] Could not resolve WFX path from log: {wfx_str}")
            continue
        outp = write_surface_json(wfx_local, sb)
        print(f"Wrote {outp}")
        written += 1

    print(f"Done. JSON written: {written}")


# -------------------------
# Dataset update
# -------------------------

def _find_best_surface_json(fragment: str, tag: str) -> Optional[Path]:
    """
    Find a JSON matching '<stem>.surface.<tag>.json' under calculations/<fragment>/**.
    We take the newest by mtime.
    """
    root = CALCULATIONS_DIR / fragment
    if not root.exists():
        return None
    candidates = list(root.rglob(f"*_InterFrag.surface.*{tag}.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _load_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = math.nan
    return df

def update_dataset(q_tag: str, overwrite: bool = False) -> None:
    if not MY_DATASET_CSV.exists():
        raise FileNotFoundError(f"Cannot find {MY_DATASET_CSV}")

    df = pd.read_csv(MY_DATASET_CSV)
    df = ensure_columns(df, ["QPSA", "QNPSA"])

    updated = 0
    skipped = 0

    for idx, row in df.iterrows():
        name = str(row.get("name", ""))
        interactor = str(row.get("interactor", "")).strip()

        if "-" not in name or not interactor:
            skipped += 1
            continue

        sugar = name.split("-", 1)[0].strip()

        # skip if already present unless overwrite
        if not overwrite and pd.notna(row.get("QPSA")) and pd.notna(row.get("QNPSA")):
            continue

        s_json = _find_best_surface_json(sugar, q_tag)
        i_json = _find_best_surface_json(interactor, q_tag)

        if not s_json or not i_json:
            skipped += 1
            continue

        s = _load_json(s_json)
        i = _load_json(i_json)

        # Polar/nonpolar areas are the QMPSA-like outputs in Multiwfn summary
        s_p = s.get("polar_area_A2")
        s_np = s.get("nonpolar_area_A2")
        i_p = i.get("polar_area_A2")
        i_np = i.get("nonpolar_area_A2")

        if any(v is None for v in (s_p, s_np, i_p, i_np)):
            skipped += 1
            continue

        df.at[idx, "QPSA"] = float(s_p) + float(i_p)
        df.at[idx, "QNPSA"] = float(s_np) + float(i_np)
        updated += 1

    df.to_csv(MY_DATASET_CSV, index=False)
    print(f"Updated rows: {updated} | skipped: {skipped}")
    print(f"Wrote {MY_DATASET_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", nargs="+", help="Path(s) to Multiwfn terminal log text files")
    ap.add_argument("--update", action="store_true", help="Update my_dataset.csv")
    ap.add_argument("--q-tag", default="func25_iso0.001", help="Surface tag to use for QPSA/QNPSA (default: func25_iso0.001)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing QPSA/QNPSA values")
    args = ap.parse_args()

    if args.ingest:
        ingest_logs([Path(p) for p in args.ingest])

    if args.update:
        update_dataset(q_tag=args.q_tag, overwrite=args.overwrite)

    if not args.ingest and not args.update:
        ap.print_help()

if __name__ == "__main__":
    main()
