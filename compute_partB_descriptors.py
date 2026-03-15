#!/usr/bin/env python3
"""
compute_partB_descriptors.py

Part-B descriptor pipeline for the Nature Chemistry 2021 paper (s41557-021-00646-w):
- Ingest raw terminal logs from Multiwfn "Quantitative analysis of molecular surface"
  (module 12) and store them as structured JSON files next to each fragment.
- Combine sugar/interactor fragment descriptors by simple mathematical operations
  to generate subsystem ("system") descriptors and update my_dataset.csv.

Designed to mirror the workflow style of compute_homo_lumo.py.  fileciteturn54file0

-------------------------------------------------------------------------
Expected project layout (same idea as compute_homo_lumo.py)
-------------------------------------------------------------------------
DP-Regenrate-Results/
  my_dataset.csv
  calculations/
    GlcNAc/
      GlcNAc-2/
        GlcNAc_InterFrag.wfx
        multiwfn_surface.log               # raw log (optional)
        GlcNAc_InterFrag.surface.json      # generated (step 1)
    Ala/
      Ala-2/
        Ala_InterFrag.wfx
        ...

You can keep multiple runs (Ala-1, Ala-2, ...). The script will pick
the newest available *.surface.json for each fragment.

-------------------------------------------------------------------------
Step 1: Generate Multiwfn log (recommended naming)
-------------------------------------------------------------------------
Run Multiwfn interactively, or redirect stdout to a file, e.g.:

  Multiwfn Ala_InterFrag.wfx | tee multiwfn_surface_iso0.001.log

The script can ingest either:
- one combined log containing multiple fragments, OR
- per-folder logs.

-------------------------------------------------------------------------
Step 2: Ingest + Update dataset
-------------------------------------------------------------------------
  python compute_partB_descriptors.py --ingest Multiwfn-Results.txt
  python compute_partB_descriptors.py --update

You can do both in one run:
  python compute_partB_descriptors.py --ingest Multiwfn-Results.txt --update

-------------------------------------------------------------------------
What gets filled
-------------------------------------------------------------------------
This script fills the Part-B columns you listed (except experimental targets):
- PSAs*NSAi, PSAs+NSAi, PSAs*NSAi/overall surface, NSAs*PSAi, NSAs+PSAi
- Volume, Minimal value, Maximal value, Overall surface area, Positive surface area,
  Negative surface area, Overall/Pos/Neg average value, Overall/Pos/Neg variance,
  miu (nu), sigma^2_tot*miu, Pi
- qPSA, qNPSA, Total Surface area
- MinESP, MaxESP

Important:
- ddGglyc and ddGglyc error are experimental; this script does not compute them.

-------------------------------------------------------------------------
Conventions (mapping to Multiwfn output fields)
-------------------------------------------------------------------------
From Multiwfn "Summary of surface analysis":

- Volume                 -> "Volume: ... ( X Angstrom^3)"
- Overall surface area   -> "... ( X Angstrom^2)"
- Positive/Negative surface area -> "... ( X Angstrom^2)"
- Minimal/Maximal value  -> in kcal/mol
- Averages/variances     -> use kcal/mol and (kcal/mol)^2 values (the numbers in parentheses)
- miu                    -> Multiwfn prints "Balance of charges (nu)" (nu == miu in your CSV)
- Pi                     -> Multiwfn prints "Internal charge separation (Pi)" (kcal/mol in parentheses)
- MinESP/MaxESP          -> "Global surface minimum/maximum" in a.u.
- qPSA/qNPSA             -> "Polar/Nonpolar surface area" (Angstrom^2)

-------------------------------------------------------------------------
System (pair) combination rules
-------------------------------------------------------------------------
We follow the paper's described "combine fragment descriptors through mathematical operations"
workflow (Supplementary Fig. 42). fileciteturn55file5

Let sugar fragment be "s" and interactor fragment be "i".
We read fragment-level values from Multiwfn and compute:

PSAs*NSAi                 = PSAs * NSAi
PSAs+NSAi                 = PSAs + NSAi      (this is q1 in the main paper) fileciteturn55file0
NSAs*PSAi                 = NSAs * PSAi
NSAs+PSAi                 = NSAs + PSAi

Overall surface area       = SAs + SAi
Positive surface area      = PSAs + NSAi     (matches q1 definition in the paper) fileciteturn55file0
Negative surface area      = Overall surface area - Positive surface area

Volume                     = Vol_s + Vol_i

For scalar surface stats (avg/var/miu/Pi/etc.) we use simple sums
to stay within "mathematical operations" (no additional assumptions):
Overall average value       = avg_s + avg_i
Positive average value      = avgpos_s + avgpos_i
Negative average value      = avgneg_s + avgneg_i
Overall variance            = var_s + var_i
Positive variance           = varpos_s + varpos_i
Negative variance           = varneg_s + varneg_i
miu                         = miu_s + miu_i
sigma^2_tot*miu             = (sigma^2_tot*miu)_s + (sigma^2_tot*miu)_i
Pi                          = Pi_s + Pi_i

qPSA/qNPSA/Total surface:
qPSA                        = qPSA_s + qPSA_i
qNPSA                       = qNPSA_s + qNPSA_i
Total Surface area          = qPSA + qNPSA

MinESP                      = min(MinESP_s, MinESP_i)
MaxESP                      = max(MaxESP_s, MaxESP_i)
Minimal value               = min(MinVal_s, MinVal_i)
Maximal value               = max(MaxVal_s, MaxVal_i)

-------------------------------------------------------------------------
Notes on matching the published dataset
-------------------------------------------------------------------------
To reproduce the exact published dataset values, fragment definition matters:
the Supplementary indicates that the interactor side chain is truncated at the Cα–Cβ bond
before QM, i.e., you should not use the full amino acid if you want to match their numbers. fileciteturn55file5

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd


# -------------------------
# Configuration
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
CALCULATIONS_DIR = BASE_DIR / "calculations"
MY_DATASET_CSV = BASE_DIR / "my_dataset.csv"

# Output JSON suffix
JSON_SUFFIX = ".surface.json"


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# -------------------------
# Parsing Multiwfn logs
# -------------------------

@dataclass
class SurfaceMetrics:
    # Units are explicitly stored as:
    # - Angstrom^3 for volume, Angstrom^2 for areas
    # - kcal/mol for values, (kcal/mol)^2 for variances
    surface_def: Optional[str] = None
    iso: Optional[float] = None

    volume_A3: Optional[float] = None
    overall_surface_A2: Optional[float] = None
    positive_surface_A2: Optional[float] = None
    negative_surface_A2: Optional[float] = None

    minimal_value_kcal: Optional[float] = None
    maximal_value_kcal: Optional[float] = None

    overall_avg_kcal: Optional[float] = None
    positive_avg_kcal: Optional[float] = None
    negative_avg_kcal: Optional[float] = None

    var_tot_kcal2: Optional[float] = None
    var_pos_kcal2: Optional[float] = None
    var_neg_kcal2: Optional[float] = None

    miu: Optional[float] = None
    prod_var_miu_kcal2: Optional[float] = None
    pi_kcal: Optional[float] = None

    minESP_au: Optional[float] = None
    maxESP_au: Optional[float] = None

    polar_area_A2: Optional[float] = None      # |ESP| > threshold
    nonpolar_area_A2: Optional[float] = None   # |ESP| <= threshold
    polar_threshold_kcal: Optional[float] = None  # threshold used by Multiwfn (default 10)


def _get_float(pattern: str, s: str) -> Optional[float]:
    m = re.search(pattern, s, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_multiwfn_surface_from_text(text: str) -> List[Tuple[str, SurfaceMetrics]]:
    """
    Parse one text blob that may contain multiple Multiwfn sessions.
    Returns a list of (loaded_wfx_path, SurfaceMetrics), keeping the last
    "Summary of surface analysis" within each session.
    """
    load_re = re.compile(r"Loaded\s+(.+?)\s+successfully!", re.MULTILINE)
    matches = list(load_re.finditer(text))
    results: List[Tuple[str, SurfaceMetrics]] = []

    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        wfx_path = m.group(1).strip()

        # Surface definition line in module header (if present)
        surf_def = None
        iso = None
        m2 = re.search(r"Select the way to define surface,\s*current:\s*([A-Za-z \-]+),\s*iso:\s*([0-9.]+)", block)
        if m2:
            surf_def = m2.group(1).strip()
            iso = float(m2.group(2))

        # Global min/max ESP (a.u.)
        minESP = _get_float(r"Global surface minimum:\s*([-\d.]+)\s*a\.u\.", block)
        maxESP = _get_float(r"Global surface maximum:\s*([-\d.]+)\s*a\.u\.", block)

        # Find the LAST summary in this session
        sum_re = re.compile(r"=+\s*Summary of surface analysis\s*=+", re.IGNORECASE)
        sums = list(sum_re.finditer(block))
        if not sums:
            continue
        sstart = sums[-1].start()
        send = block.find("Surface analysis finished!", sstart)
        if send == -1:
            send = len(block)
        summary = block[sstart:send]

        metrics = SurfaceMetrics(surface_def=surf_def, iso=iso, minESP_au=minESP, maxESP_au=maxESP)

        metrics.volume_A3 = _get_float(r"Volume:\s*[-\d.]+\s*Bohr\^3\s*\(\s*([-\d.]+)\s*Angstrom\^3\)", summary)
        metrics.minimal_value_kcal = _get_float(r"Minimal value:\s*([-\d.]+)\s*kcal/mol", summary)
        metrics.maximal_value_kcal = _get_float(r"Maximal value:\s*([-\d.]+)\s*kcal/mol", summary)

        metrics.overall_surface_A2 = _get_float(r"Overall surface area:\s*[-\d.]+\s*Bohr\^2\s*\(\s*([-\d.]+)\s*Angstrom\^2\)", summary)
        metrics.positive_surface_A2 = _get_float(r"Positive surface area:\s*[-\d.]+\s*Bohr\^2\s*\(\s*([-\d.]+)\s*Angstrom\^2\)", summary)
        metrics.negative_surface_A2 = _get_float(r"Negative surface area:\s*[-\d.]+\s*Bohr\^2\s*\(\s*([-\d.]+)\s*Angstrom\^2\)", summary)

        metrics.overall_avg_kcal = _get_float(r"Overall average value:\s*[-\d.]+\s*a\.u\.\s*\(\s*([-\d.]+)\s*kcal/mol\)", summary)
        metrics.positive_avg_kcal = _get_float(r"Positive average value:\s*[-\d.]+\s*a\.u\.\s*\(\s*([-\d.]+)\s*kcal/mol\)", summary)
        metrics.negative_avg_kcal = _get_float(r"Negative average value:\s*[-\d.]+\s*a\.u\.\s*\(\s*([-\d.]+)\s*kcal/mol\)", summary)

        metrics.var_tot_kcal2 = _get_float(r"Overall variance \(sigma\^2_tot\):\s*[-\d.]+\s*a\.u\.\^2\s*\(\s*([-\d.]+)\s*\(kcal/mol\)\^2\)", summary)
        metrics.var_pos_kcal2 = _get_float(r"Positive variance:\s*[-\d.]+\s*a\.u\.\^2\s*\(\s*([-\d.]+)\s*\(kcal/mol\)\^2\)", summary)
        metrics.var_neg_kcal2 = _get_float(r"Negative variance:\s*[-\d.]+\s*a\.u\.\^2\s*\(\s*([-\d.]+)\s*\(kcal/mol\)\^2\)", summary)

        metrics.miu = _get_float(r"Balance of charges \(nu\):\s*([-\d.]+)", summary)
        metrics.prod_var_miu_kcal2 = _get_float(r"Product of sigma\^2_tot and nu:\s*[-\d.]+\s*a\.u\.\^2\s*\(\s*([-\d.]+)\s*\(kcal/mol\)\^2\)", summary)
        metrics.pi_kcal = _get_float(r"Internal charge separation \(Pi\):\s*[-\d.]+\s*a\.u\.\s*\(\s*([-\d.]+)\s*kcal/mol\)", summary)

        # Polar / nonpolar by |ESP| threshold (kcal/mol) reported by Multiwfn
        mnp = re.search(r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\):\s*([-\d.]+)\s*Angstrom\^2", summary)
        if mnp:
            metrics.polar_threshold_kcal = float(mnp.group(1))
            metrics.nonpolar_area_A2 = float(mnp.group(2))
        else:
            metrics.nonpolar_area_A2 = _get_float(r"Nonpolar surface area.*?:\s*([-\d.]+)\s*Angstrom\^2", summary)
            metrics.polar_threshold_kcal = 10.0

        mp = re.search(r"Polar surface area\s*\(\|ESP\|\s*>\s*[0-9.]+\s*kcal/mol\):\s*([-\d.]+)\s*Angstrom\^2", summary)
        if mp:
            metrics.polar_area_A2 = float(mp.group(1))
        else:
            metrics.polar_area_A2 = _get_float(r"Polar surface area.*?:\s*([-\d.]+)\s*Angstrom\^2", summary)

        results.append((wfx_path, metrics))

    return results


def write_surface_json(next_to_wfx: Path, metrics: SurfaceMetrics) -> Path:
    """
    Write metrics to <basename>.surface.json next to the wfx file.
    """
    out_path = next_to_wfx.with_suffix(next_to_wfx.suffix + JSON_SUFFIX)
    out_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return out_path


def _choose_best_wfx_candidate(wfx_hint_path: str, candidates: List[Path]) -> Optional[Path]:
    """
    Pick the best candidate .wfx path under calculations/ from a hint path string
    found in Multiwfn logs.
    """
    if not candidates:
        return None

    hint = wfx_hint_path.replace("\\", "/")
    hint_name = Path(hint).name
    hint_parent = Path(hint).parent.name

    # 1) exact basename match
    by_name = [p for p in candidates if p.name == hint_name]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        # Prefer a directory name that appears in the hint path (e.g. Ala-2)
        if hint_parent:
            ranked = [p for p in by_name if p.parent.name == hint_parent]
            if ranked:
                ranked.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return ranked[0]
        by_name.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return by_name[0]

    # 2) fuzzy path overlap
    hint_norm = _norm_name(hint)
    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        p_norm = _norm_name(str(p))
        score = 0
        if _norm_name(p.name) in hint_norm:
            score += 5
        if _norm_name(p.parent.name) in hint_norm:
            score += 2
        scored.append((score, p))
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]


def ingest_log_file(log_path: Path) -> int:
    """
    Parse one Multiwfn .log file and write surface JSON files next to the
    corresponding .wfx file(s).

    Returns the number of JSON files written.
    """
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_multiwfn_surface_from_text(txt)
    if not parsed:
        print(f"WARNING: no Multiwfn surface sessions found in {log_path}")
        return 0

    # Keep the last parsed metrics per loaded path inside this log
    by_wfx: Dict[str, SurfaceMetrics] = {}
    for wfx_path, metrics in parsed:
        by_wfx[wfx_path] = metrics

    all_candidates = list(CALCULATIONS_DIR.rglob("*.wfx"))
    written = 0

    for wfx_path, metrics in by_wfx.items():
        target_wfx = _choose_best_wfx_candidate(wfx_path, all_candidates)
        if target_wfx is None:
            print(f"WARNING: cannot map loaded path from log to local .wfx: {wfx_path}")
            continue
        out_json = write_surface_json(target_wfx, metrics)
        written += 1
        print(f"Wrote {out_json.relative_to(BASE_DIR)}  <-  {log_path.relative_to(BASE_DIR)}")

    return written


def _log_source_priority(log_path: Path) -> int:
    """
    Rank log types for Part-B extraction.
    Higher score means more preferred.
    """
    name = log_path.name.lower()
    # Prefer electron-density iso(0.001) logs for Part-B descriptors.
    if "iso" in name and "vdw" not in name:
        return 3
    if "vdw" in name:
        return 2
    return 1


def _parse_log_to_target_metrics(log_path: Path) -> Dict[Path, SurfaceMetrics]:
    """
    Parse one log and map it to target local .wfx files.
    Returns {target_wfx_path: SurfaceMetrics}.
    """
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_multiwfn_surface_from_text(txt)
    if not parsed:
        print(f"WARNING: no Multiwfn surface sessions found in {log_path}")
        return {}

    by_wfx: Dict[str, SurfaceMetrics] = {}
    for wfx_path, metrics in parsed:
        by_wfx[wfx_path] = metrics

    all_candidates = list(CALCULATIONS_DIR.rglob("*.wfx"))
    out: Dict[Path, SurfaceMetrics] = {}
    for wfx_path, metrics in by_wfx.items():
        target_wfx = _choose_best_wfx_candidate(wfx_path, all_candidates)
        if target_wfx is None:
            print(f"WARNING: cannot map loaded path from log to local .wfx: {wfx_path}")
            continue
        out[target_wfx] = metrics

    return out


def discover_surface_logs() -> List[Path]:
    """
    Discover candidate Multiwfn surface logs under calculations/.
    """
    patterns = [
        "**/multiwfn_surface*.log",
        "**/*surface*.log",
    ]
    seen = set()
    logs: List[Path] = []
    for pat in patterns:
        for p in CALCULATIONS_DIR.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                logs.append(p)
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


def ingest_logs(paths: List[Path]) -> int:
    """
    Ingest a list of log files. Returns total JSON files written.
    """
    # Collect candidates and pick one best source log per target .wfx file.
    selected: Dict[Path, Tuple[int, float, Path, SurfaceMetrics]] = {}

    for pth in paths:
        if not pth.exists():
            print(f"WARNING: ingest file not found: {pth}")
            continue
        if pth.suffix.lower() != ".log":
            print(f"WARNING: skipping non-log file: {pth}")
            continue

        per_target = _parse_log_to_target_metrics(pth)
        prio = _log_source_priority(pth)
        mtime = pth.stat().st_mtime

        for target_wfx, metrics in per_target.items():
            cand = (prio, mtime, pth, metrics)
            prev = selected.get(target_wfx)
            if prev is None or (cand[0], cand[1]) > (prev[0], prev[1]):
                selected[target_wfx] = cand

    written = 0
    for target_wfx, (_prio, _mtime, src_log, metrics) in selected.items():
        out_json = write_surface_json(target_wfx, metrics)
        written += 1
        print(f"Wrote {out_json.relative_to(BASE_DIR)}  <-  {src_log.relative_to(BASE_DIR)}")

    return written


# -------------------------
# Locate fragment results
# -------------------------

def newest_surface_json_for_fragment(fragment_name: str) -> Optional[Path]:
    """
    Find the newest *.surface.json under calculations/<fragment_name>/.
    """
    frag_dir = CALCULATIONS_DIR / fragment_name
    if not frag_dir.is_dir():
        return None

    candidates = list(frag_dir.rglob(f"*{JSON_SUFFIX}"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_surface_metrics(json_path: Path) -> SurfaceMetrics:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return SurfaceMetrics(**data)


# -------------------------
# System descriptor computation
# -------------------------

PARTB_COLS = [
    "PSAs*NSAi", "PSAs+NSAi", "PSAs*NSAi/overall surface", "NSAs*PSAi", "NSAs+PSAi",
    "Volume", "Minimal value", "Maximal value",
    "Overall surface area", "Positive surface area", "Negative surface area",
    "Overall average value", "Positive average value", "Negative average value",
    "Overall variance (sigma^2_tot)", "Positive variance", "Negative variance",
    "Balance of charges (miu)", "Product of sigma^2_tot and miu", "Internal charge separation (Pi)",
    "qNPSA", "qPSA", "Total Surface area", "MinESP", "MaxESP",
    # Uppercase forms are left for a dedicated vdW run, but we keep columns:
    "QNPSA", "QPSA",
]


def compute_partB_from_fragments(s: SurfaceMetrics, i: SurfaceMetrics) -> Dict[str, float]:
    """
    Compute system descriptors from sugar (s) and interactor (i) fragment metrics.
    """
    # shorthand
    PSAs = s.positive_surface_A2
    NSAs = s.negative_surface_A2
    PSAi = i.positive_surface_A2
    NSAi = i.negative_surface_A2

    # Safety
    if any(v is None for v in (PSAs, NSAs, PSAi, NSAi)):
        raise ValueError("Missing Positive/Negative surface area in one of the fragments.")

    out: Dict[str, float] = {}

    out["PSAs*NSAi"] = PSAs * NSAi
    out["PSAs+NSAi"] = PSAs + NSAi  # q1 fileciteturn55file0
    out["NSAs*PSAi"] = NSAs * PSAi
    out["NSAs+PSAi"] = NSAs + PSAi

    # Surface areas
    SA_total = (s.overall_surface_A2 or 0.0) + (i.overall_surface_A2 or 0.0)
    out["Overall surface area"] = SA_total
    out["Positive surface area"] = out["PSAs+NSAi"]
    out["Negative surface area"] = SA_total - out["Positive surface area"]

    # Normalized
    out["PSAs*NSAi/overall surface"] = out["PSAs*NSAi"] / SA_total if SA_total else float("nan")

    # Volume
    out["Volume"] = (s.volume_A3 or 0.0) + (i.volume_A3 or 0.0)

    # Min/Max values and ESP extrema
    out["Minimal value"] = min(s.minimal_value_kcal, i.minimal_value_kcal) if (s.minimal_value_kcal is not None and i.minimal_value_kcal is not None) else float("nan")
    out["Maximal value"] = max(s.maximal_value_kcal, i.maximal_value_kcal) if (s.maximal_value_kcal is not None and i.maximal_value_kcal is not None) else float("nan")
    out["MinESP"] = min(s.minESP_au, i.minESP_au) if (s.minESP_au is not None and i.minESP_au is not None) else float("nan")
    out["MaxESP"] = max(s.maxESP_au, i.maxESP_au) if (s.maxESP_au is not None and i.maxESP_au is not None) else float("nan")

    # Simple sums for scalar stats (consistent with "mathematical operations") fileciteturn55file5
    def ssum(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None:
            return float("nan")
        return a + b

    out["Overall average value"] = ssum(s.overall_avg_kcal, i.overall_avg_kcal)
    out["Positive average value"] = ssum(s.positive_avg_kcal, i.positive_avg_kcal)
    out["Negative average value"] = ssum(s.negative_avg_kcal, i.negative_avg_kcal)

    out["Overall variance (sigma^2_tot)"] = ssum(s.var_tot_kcal2, i.var_tot_kcal2)
    out["Positive variance"] = ssum(s.var_pos_kcal2, i.var_pos_kcal2)
    out["Negative variance"] = ssum(s.var_neg_kcal2, i.var_neg_kcal2)

    out["Balance of charges (miu)"] = ssum(s.miu, i.miu)
    out["Product of sigma^2_tot and miu"] = ssum(s.prod_var_miu_kcal2, i.prod_var_miu_kcal2)
    out["Internal charge separation (Pi)"] = ssum(s.pi_kcal, i.pi_kcal)

    # QMPSA-like (polar/nonpolar by |ESP| threshold) from Multiwfn summary.
    # We map these to lower-case qPSA/qNPSA and "Total Surface area".
    if s.polar_area_A2 is not None and i.polar_area_A2 is not None:
        out["qPSA"] = s.polar_area_A2 + i.polar_area_A2
    else:
        out["qPSA"] = float("nan")
    if s.nonpolar_area_A2 is not None and i.nonpolar_area_A2 is not None:
        out["qNPSA"] = s.nonpolar_area_A2 + i.nonpolar_area_A2
    else:
        out["qNPSA"] = float("nan")
    if not (math.isnan(out["qPSA"]) or math.isnan(out["qNPSA"])):
        out["Total Surface area"] = out["qPSA"] + out["qNPSA"]
    else:
        out["Total Surface area"] = float("nan")

    # Uppercase QPSA/QNPSA are left as NaN unless you ingest a dedicated vdW-surface run and store it separately.
    out["QPSA"] = float("nan")
    out["QNPSA"] = float("nan")

    return out


# -------------------------
# Update my_dataset.csv
# -------------------------

def update_my_dataset(overwrite_existing: bool = True):
    if not MY_DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {MY_DATASET_CSV}")

    # compute_homo_lumo.py uses index_col=0; keep same convention fileciteturn54file0
    df = pd.read_csv(MY_DATASET_CSV, index_col=0)

    if "name" not in df.columns or "interactor" not in df.columns:
        raise ValueError("my_dataset.csv must contain 'name' and 'interactor' columns.")

    # Ensure Part-B columns exist
    for c in PARTB_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    updated = 0

    for idx, row in df.iterrows():
        name = str(row["name"])
        interactor = str(row["interactor"])
        sugar = name.split("-", 1)[0]

        sugar_json = newest_surface_json_for_fragment(sugar)
        inter_json = newest_surface_json_for_fragment(interactor)

        if sugar_json is None or inter_json is None:
            continue

        s_metrics = load_surface_metrics(sugar_json)
        i_metrics = load_surface_metrics(inter_json)

        try:
            vals = compute_partB_from_fragments(s_metrics, i_metrics)
        except Exception:
            continue

        # By default overwrite existing Part-B values to ensure consistency
        # with the latest fragment logs/JSON.
        if not overwrite_existing:
            first_col = "PSAs+NSAi"
            if pd.notna(row.get(first_col, pd.NA)):
                continue

        for k, v in vals.items():
            df.at[idx, k] = v

        updated += 1

    df.to_csv(MY_DATASET_CSV)
    print(f"Finished. Updated {updated} row(s). Saved to: {MY_DATASET_CSV}")


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", nargs="*", default=[],
                    help="Path(s) to raw Multiwfn .log file(s) to ingest.")
    ap.add_argument("--ingest-logs", action="store_true",
                    help="Auto-discover and ingest calculations/**/multiwfn_surface*.log files.")
    ap.add_argument("--update", action="store_true",
                    help="Update my_dataset.csv using the newest *.surface.json files.")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="When updating CSV, keep existing Part-B values and fill only blanks.")
    args = ap.parse_args()

    # Resolve explicit --ingest paths and keep only .log files.
    explicit_logs: List[Path] = [Path(p) for p in args.ingest]

    if args.ingest:
        written = ingest_logs(explicit_logs)
        print(f"Ingest from explicit logs complete: wrote {written} JSON file(s).")

    # If requested (or implied by --update), auto-discover .log files and ingest them.
    # This makes the pipeline log-first and avoids dependency on generic .txt files.
    if args.ingest_logs or args.update:
        auto_logs = discover_surface_logs()
        if auto_logs:
            written = ingest_logs(auto_logs)
            print(f"Auto-ingest complete: wrote {written} JSON file(s) from {len(auto_logs)} log(s).")
        else:
            print(f"WARNING: no .log files discovered under {CALCULATIONS_DIR}")

    if args.update:
        update_my_dataset(overwrite_existing=(not args.no_overwrite))


if __name__ == "__main__":
    main()
