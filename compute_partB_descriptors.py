#!/usr/bin/env python3
"""
compute_partB_descriptors.py

Part-B descriptor pipeline:
- Parse Multiwfn "Quantitative molecular surface analysis" (module 12) logs
  and write structured JSON next to each fragment's *.wfx.
- Combine sugar/interactor fragment descriptors using simple math operations
  and update my_dataset.csv.

DEFAULT (no-args) behavior (matches compute_homo_lumo.py style):
    python compute_partB_descriptors.py

It will:
  - auto-discover logs under calculations/ (multiwfn_surface*.log/.txt, *surface*.log/.txt, multiwfn*.log/.txt)
  - ingest -> write *.surface.json next to mapped *.wfx
  - update my_dataset.csv (fills blanks; skips already-computed rows)

Optional CLI:
  --ingest <files...>     ingest explicit log files (.log/.txt)
  --ingest-logs           auto-discover and ingest logs under calculations/
  --update                update my_dataset.csv from newest *.surface.json
  --overwrite             overwrite existing Part-B values (default: fill only blanks)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
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

JSON_SUFFIX = ".surface.json"


# -------------------------
# Helpers
# -------------------------

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    t = x.strip()
    if t.lower() in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(t)
    except Exception:
        return None


def _get_float(pattern: str, s: str) -> Optional[float]:
    m = re.search(pattern, s, flags=re.MULTILINE)
    if not m:
        return None
    return _to_float(m.group(1))


# -------------------------
# Parsing Multiwfn logs
# -------------------------

@dataclass
class SurfaceMetrics:
    # Stored units:
    # - volume: Angstrom^3
    # - surface areas: Angstrom^2
    # - values/averages/Pi: kcal/mol
    # - variances, sigma^2_tot*nu: (kcal/mol)^2
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

    miu: Optional[float] = None  # Multiwfn prints as "nu" (paper uses miu)
    prod_var_miu_kcal2: Optional[float] = None
    pi_kcal: Optional[float] = None

    minESP_au: Optional[float] = None
    maxESP_au: Optional[float] = None

    polar_area_A2: Optional[float] = None      # Multiwfn "Polar surface area (|ESP|>thr)"
    nonpolar_area_A2: Optional[float] = None   # Multiwfn "Nonpolar surface area (|ESP|<=thr)"
    polar_threshold_kcal: Optional[float] = None


def parse_multiwfn_surface_from_text(text: str) -> List[Tuple[str, SurfaceMetrics]]:
    """
    Parse one text blob that may contain multiple Multiwfn sessions.
    Returns a list of (loaded_wfx_path_string, SurfaceMetrics).
    """
    load_re = re.compile(r"Loaded\s+(.+?)\s+successfully!", re.MULTILINE)
    matches = list(load_re.finditer(text))
    results: List[Tuple[str, SurfaceMetrics]] = []

    if not matches:
        return results

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        block = text[start:end]
        wfx_path = m.group(1).strip()

        # surface definition line
        surf_def = None
        iso = None
        m2 = re.search(
            r"Select the way to define surface,\s*current:\s*([A-Za-z \-]+),\s*iso:\s*([0-9.]+)",
            block
        )
        if m2:
            surf_def = m2.group(1).strip()
            iso = _to_float(m2.group(2))

        # global min/max ESP (a.u.)
        minESP = _get_float(r"Global surface minimum:\s*([-\d.+eE]+|NaN)\s*a\.u\.", block)
        maxESP = _get_float(r"Global surface maximum:\s*([-\d.+eE]+|NaN)\s*a\.u\.", block)

        # last summary
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

        metrics.volume_A3 = _get_float(
            r"Volume:\s*[-\d.+eE]+\s*Bohr\^3\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^3\)",
            summary
        )

        metrics.minimal_value_kcal = _get_float(r"Minimal value:\s*([-\d.+eE]+|NaN)\s*kcal/mol", summary)
        metrics.maximal_value_kcal = _get_float(r"Maximal value:\s*([-\d.+eE]+|NaN)\s*kcal/mol", summary)

        metrics.overall_surface_A2 = _get_float(
            r"Overall surface area:\s*[-\d.+eE]+\s*Bohr\^2\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^2\)",
            summary
        )
        metrics.positive_surface_A2 = _get_float(
            r"Positive surface area:\s*[-\d.+eE]+\s*Bohr\^2\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^2\)",
            summary
        )
        metrics.negative_surface_A2 = _get_float(
            r"Negative surface area:\s*[-\d.+eE]+\s*Bohr\^2\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^2\)",
            summary
        )

        metrics.overall_avg_kcal = _get_float(
            r"Overall average value:\s*[-\d.+eE]+|NaN\s*a\.u\.\s*\(\s*([-\d.+eE]+|NaN)\s*kcal/mol\)",
            summary
        )
        # Some Multiwfn builds print NaN directly; keep regex robust:
        metrics.overall_avg_kcal = _get_float(
            r"Overall average value:\s*([-\d.+eE]+|NaN)\s*a\.u\.\s*\(\s*([-\d.+eE]+|NaN)\s*kcal/mol\)",
            summary
        ) if metrics.overall_avg_kcal is None else metrics.overall_avg_kcal

        metrics.positive_avg_kcal = _get_float(
            r"Positive average value:\s*([-\d.+eE]+|NaN)\s*a\.u\.\s*\(\s*([-\d.+eE]+|NaN)\s*kcal/mol\)",
            summary
        )
        metrics.negative_avg_kcal = _get_float(
            r"Negative average value:\s*([-\d.+eE]+|NaN)\s*a\.u\.\s*\(\s*([-\d.+eE]+|NaN)\s*kcal/mol\)",
            summary
        )

        metrics.var_tot_kcal2 = _get_float(
            r"Overall variance \(sigma\^2_tot\):\s*([-\d.+eE]+|NaN)\s*a\.u\.\^2\s*\(\s*([-\d.+eE]+|NaN)\s*\(kcal/mol\)\^2\)",
            summary
        )
        metrics.var_pos_kcal2 = _get_float(
            r"Positive variance:\s*([-\d.+eE]+|NaN)\s*a\.u\.\^2\s*\(\s*([-\d.+eE]+|NaN)\s*\(kcal/mol\)\^2\)",
            summary
        )
        metrics.var_neg_kcal2 = _get_float(
            r"Negative variance:\s*([-\d.+eE]+|NaN)\s*a\.u\.\^2\s*\(\s*([-\d.+eE]+|NaN)\s*\(kcal/mol\)\^2\)",
            summary
        )

        metrics.miu = _get_float(r"Balance of charges \(nu\):\s*([-\d.+eE]+|NaN)", summary)
        metrics.prod_var_miu_kcal2 = _get_float(
            r"Product of sigma\^2_tot and nu:\s*([-\d.+eE]+|NaN)\s*a\.u\.\^2\s*\(\s*([-\d.+eE]+|NaN)\s*\(kcal/mol\)\^2\)",
            summary
        )
        metrics.pi_kcal = _get_float(
            r"Internal charge separation \(Pi\):\s*([-\d.+eE]+|NaN)\s*a\.u\.\s*\(\s*([-\d.+eE]+|NaN)\s*kcal/mol\)",
            summary
        )

        # Polar / nonpolar by |ESP| threshold
        mnp = re.search(
            r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\):\s*([-\d.+eE]+|NaN)\s*Angstrom\^2",
            summary
        )
        if mnp:
            metrics.polar_threshold_kcal = _to_float(mnp.group(1))
            metrics.nonpolar_area_A2 = _to_float(mnp.group(2))
        else:
            metrics.polar_threshold_kcal = 10.0
            metrics.nonpolar_area_A2 = _get_float(r"Nonpolar surface area.*?:\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", summary)

        mp = re.search(
            r"Polar surface area\s*\(\|ESP\|\s*>\s*[0-9.]+\s*kcal/mol\):\s*([-\d.+eE]+|NaN)\s*Angstrom\^2",
            summary
        )
        if mp:
            metrics.polar_area_A2 = _to_float(mp.group(1))
        else:
            metrics.polar_area_A2 = _get_float(r"Polar surface area.*?:\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", summary)

        results.append((wfx_path, metrics))

    return results


def write_surface_json(next_to_wfx: Path, metrics: SurfaceMetrics) -> Path:
    out_path = next_to_wfx.with_suffix(next_to_wfx.suffix + JSON_SUFFIX)
    out_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return out_path


def _choose_best_wfx_candidate(wfx_hint_path: str, candidates: List[Path]) -> Optional[Path]:
    """
    Map loaded path from Multiwfn log to a local *.wfx under calculations/.
    """
    if not candidates:
        return None

    hint = (wfx_hint_path or "").replace("\\", "/")
    hint_name = Path(hint).name
    hint_parent = Path(hint).parent.name

    # 1) exact basename match
    by_name = [p for p in candidates if p.name == hint_name]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        if hint_parent:
            ranked = [p for p in by_name if p.parent.name == hint_parent]
            if ranked:
                ranked.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return ranked[0]
        by_name.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return by_name[0]

    # 2) fuzzy overlap
    hint_norm = _norm_name(hint)
    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        p_norm = _norm_name(str(p))
        score = 0
        if _norm_name(p.name) and _norm_name(p.name) in hint_norm:
            score += 5
        if _norm_name(p.parent.name) and _norm_name(p.parent.name) in hint_norm:
            score += 2
        if p_norm and p_norm in hint_norm:
            score += 1
        scored.append((score, p))
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1] if scored else None


def _log_source_priority(log_path: Path) -> int:
    """
    Prefer iso(0.001) logs for Part-B.
    """
    name = log_path.name.lower()
    if "iso" in name and "vdw" not in name:
        return 3
    if "vdw" in name:
        return 2
    return 1


def _parse_log_to_target_metrics(log_path: Path) -> Dict[Path, SurfaceMetrics]:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_multiwfn_surface_from_text(txt)
    if not parsed:
        return {}

    # last metrics per loaded path
    by_wfx: Dict[str, SurfaceMetrics] = {}
    for wfx_path, metrics in parsed:
        by_wfx[wfx_path] = metrics

    all_candidates = list(CALCULATIONS_DIR.rglob("*.wfx"))
    out: Dict[Path, SurfaceMetrics] = {}
    for wfx_path, metrics in by_wfx.items():
        target_wfx = _choose_best_wfx_candidate(wfx_path, all_candidates)
        if target_wfx is None:
            continue
        out[target_wfx] = metrics
    return out


def discover_surface_logs() -> List[Path]:
    """
    Discover candidate Multiwfn surface logs under calculations/.
    Supports .log and .txt (because many people do `tee ...txt`).
    """
    patterns = [
        "**/multiwfn_surface*.log",
        "**/multiwfn_surface*.txt",
        "**/*surface*.log",
        "**/*surface*.txt",
        "**/multiwfn*.log",
        "**/multiwfn*.txt",
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
    Ingest log files and write *.surface.json next to mapped *.wfx.
    Picks best log per target *.wfx (iso > vdw > other, newest wins).
    """
    selected: Dict[Path, Tuple[int, float, Path, SurfaceMetrics]] = {}

    for pth in paths:
        if not pth.exists() or not pth.is_file():
            continue
        if pth.suffix.lower() not in {".log", ".txt"}:
            continue

        per_target = _parse_log_to_target_metrics(pth)
        if not per_target:
            continue

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
        print(f"Wrote {out_json.relative_to(BASE_DIR)} <- {src_log.relative_to(BASE_DIR)}")

    return written


# -------------------------
# Locate fragment results (JSON)
# -------------------------

def newest_surface_json_for_fragment(fragment_name: str) -> Optional[Path]:
    frag_dir = CALCULATIONS_DIR / fragment_name
    if not frag_dir.is_dir():
        return None
    candidates = list(frag_dir.rglob(f"*{JSON_SUFFIX}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_surface_metrics(json_path: Path) -> SurfaceMetrics:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
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
    "QNPSA", "QPSA",
]


def compute_partB_from_fragments(s: SurfaceMetrics, i: SurfaceMetrics) -> Dict[str, float]:
    PSAs = s.positive_surface_A2
    NSAs = s.negative_surface_A2
    PSAi = i.positive_surface_A2
    NSAi = i.negative_surface_A2

    if any(v is None for v in (PSAs, NSAs, PSAi, NSAi)):
        raise ValueError("Missing Positive/Negative surface area in one of the fragments.")

    out: Dict[str, float] = {}
    out["PSAs*NSAi"] = PSAs * NSAi
    out["PSAs+NSAi"] = PSAs + NSAi
    out["NSAs*PSAi"] = NSAs * PSAi
    out["NSAs+PSAi"] = NSAs + PSAi

    SA_total = (s.overall_surface_A2 or 0.0) + (i.overall_surface_A2 or 0.0)
    out["Overall surface area"] = SA_total
    out["Positive surface area"] = out["PSAs+NSAi"]
    out["Negative surface area"] = SA_total - out["Positive surface area"]

    out["PSAs*NSAi/overall surface"] = out["PSAs*NSAi"] / SA_total if SA_total else float("nan")
    out["Volume"] = (s.volume_A3 or 0.0) + (i.volume_A3 or 0.0)

    # extrema: pick the most extreme across fragments
    out["Minimal value"] = min(s.minimal_value_kcal, i.minimal_value_kcal) if (s.minimal_value_kcal is not None and i.minimal_value_kcal is not None) else float("nan")
    out["Maximal value"] = max(s.maximal_value_kcal, i.maximal_value_kcal) if (s.maximal_value_kcal is not None and i.maximal_value_kcal is not None) else float("nan")
    out["MinESP"] = min(s.minESP_au, i.minESP_au) if (s.minESP_au is not None and i.minESP_au is not None) else float("nan")
    out["MaxESP"] = max(s.maxESP_au, i.maxESP_au) if (s.maxESP_au is not None and i.maxESP_au is not None) else float("nan")

    def ssum(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None:
            return float("nan")
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
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

    # Multiwfn polar/nonpolar (|ESP| threshold)
    out["qPSA"] = ssum(s.polar_area_A2, i.polar_area_A2)
    out["qNPSA"] = ssum(s.nonpolar_area_A2, i.nonpolar_area_A2)
    out["Total Surface area"] = out["qPSA"] + out["qNPSA"] if not (math.isnan(out["qPSA"]) or math.isnan(out["qNPSA"])) else float("nan")

    # Uppercase QPSA/QNPSA reserved for QMPSA (Molden) workflow
    out["QPSA"] = float("nan")
    out["QNPSA"] = float("nan")

    return out


# -------------------------
# Update my_dataset.csv
# -------------------------

def update_my_dataset(fill_only_blanks: bool = True) -> int:
    if not MY_DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {MY_DATASET_CSV}")

    df = pd.read_csv(MY_DATASET_CSV, index_col=0)

    if "name" not in df.columns or "interactor" not in df.columns:
        raise ValueError("my_dataset.csv must contain 'name' and 'interactor' columns.")

    for c in PARTB_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    first_col = "PSAs+NSAi"
    updated = 0

    for idx, row in df.iterrows():
        name = str(row["name"])
        interactor = str(row["interactor"])
        sugar = name.split("-", 1)[0]

        if fill_only_blanks:
            val = row.get(first_col, pd.NA)
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                print(f"[{idx}] {name}: already computed — skipping.")
                continue

        sugar_json = newest_surface_json_for_fragment(sugar)
        inter_json = newest_surface_json_for_fragment(interactor)

        if sugar_json is None:
            print(f"[{idx}] {name}: surface JSON not found for sugar '{sugar}' — skipping.")
            continue
        if inter_json is None:
            print(f"[{idx}] {name}: surface JSON not found for interactor '{interactor}' — skipping.")
            continue

        try:
            s_metrics = load_surface_metrics(sugar_json)
            i_metrics = load_surface_metrics(inter_json)
            vals = compute_partB_from_fragments(s_metrics, i_metrics)
        except Exception as exc:
            print(f"[{idx}] {name}: ERROR {exc} — skipping.")
            continue

        print(f"[{idx}] {name}: computed Part-B descriptors.")
        for k, v in vals.items():
            df.at[idx, k] = v

        updated += 1

    df.to_csv(MY_DATASET_CSV)
    print(f"\nFinished. Updated {updated} row(s). Saved to: {MY_DATASET_CSV}")
    return updated


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", nargs="*", default=[],
                    help="Path(s) to raw Multiwfn surface log(s) to ingest (.log/.txt).")
    ap.add_argument("--ingest-logs", action="store_true",
                    help="Auto-discover and ingest logs under calculations/.")
    ap.add_argument("--update", action="store_true",
                    help="Update my_dataset.csv using newest *.surface.json.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing Part-B values (default: fill only blanks).")
    args = ap.parse_args()

    # No-args mode: behave like compute_homo_lumo.py (auto ingest + update; fill blanks)
    if len(sys.argv) == 1:
        args.ingest_logs = True
        args.update = True
        args.overwrite = False

    total_written = 0

    # Explicit ingest
    if args.ingest:
        paths = [Path(p) for p in args.ingest]
        w = ingest_logs(paths)
        total_written += w
        print(f"Ingest complete: wrote {w} JSON file(s) from explicit paths.")

    # Auto ingest (recommended)
    if args.ingest_logs or args.update:
        logs = discover_surface_logs()
        if logs:
            w = ingest_logs(logs)
            total_written += w
            print(f"Auto-ingest complete: wrote {w} JSON file(s) from {len(logs)} log(s).")
        else:
            print(f"WARNING: no Multiwfn surface logs found under {CALCULATIONS_DIR}")

    # Update CSV
    if args.update:
        update_my_dataset(fill_only_blanks=(not args.overwrite))


if __name__ == "__main__":
    main()