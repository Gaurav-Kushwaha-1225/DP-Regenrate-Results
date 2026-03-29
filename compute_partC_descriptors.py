#!/usr/bin/env python3
"""
compute_partC_descriptors.py

Run with no args:
    python compute_partC_descriptors.py

Behavior:
  - auto-discover Multiwfn logs under calculations/** (.log/.txt)
  - parse ALL Multiwfn module-12 surface summaries in those logs
  - write structured JSON per fragment + per surface tag:
        <frag>.wfx.surface.<tag>.json
  - update my_dataset.csv:
        QPSA, QNPSA
    using the best available tag per fragment.

Tag selection priority (default):
    func25_iso0.001  >  vdw_iso0.001  >  edens_iso0.001  >  newest-any

Notes:
- "Polar surface area" and "Nonpolar surface area" in Multiwfn summary are used as PSA/NPSA for that surface run.
- Part-C fills uppercase QPSA/QNPSA only. (Part-B already fills lowercase qPSA/qNPSA.)

Optional:
  --ingest <files...>     ingest explicit logs
  --ingest-logs           auto-discover logs
  --update                update my_dataset.csv
  --overwrite             overwrite existing QPSA/QNPSA (default fills blanks)
  --tag <tag>             force a specific tag (e.g., func25_iso0.001)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CALCULATIONS_DIR = BASE_DIR / "calculations"
MY_DATASET_CSV = BASE_DIR / "my_dataset.csv"

JSON_SUFFIX = ".json"


# -------------------------
# helpers
# -------------------------

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


def _fmt_iso(iso: Optional[float]) -> str:
    if iso is None:
        return "unknown"
    s = f"{iso:.6f}".rstrip("0").rstrip(".")
    return s


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip()


# -------------------------
# data model
# -------------------------

@dataclass
class PartCSurfaceMetrics:
    tag: str
    surface_def: Optional[str] = None
    iso: Optional[float] = None

    # Å^2
    total_surface_A2: Optional[float] = None
    polar_area_A2: Optional[float] = None
    nonpolar_area_A2: Optional[float] = None

    polar_threshold_kcal: Optional[float] = None

    # optional, kept for debugging
    volume_A3: Optional[float] = None
    minESP_au: Optional[float] = None
    maxESP_au: Optional[float] = None


# -------------------------
# parsing Multiwfn logs (multiple summaries per session)
# -------------------------

LOAD_RE = re.compile(r"Loaded\s+(.+?)\s+successfully!", re.MULTILINE)
SUMMARY_RE = re.compile(r"=+\s*Summary of surface analysis\s*=+", re.IGNORECASE)
FINISH_RE = re.compile(r"Surface analysis finished!", re.IGNORECASE)

# Most common line in module-12 menu:
SURF_LINE_RE = re.compile(
    r"Select the way to define surface,\s*current:\s*([A-Za-z \-]+),\s*iso:\s*([0-9.]+)",
    re.IGNORECASE
)

def _detect_surface_tag(segment: str, surface_def: Optional[str], iso: Optional[float]) -> str:
    seg_l = segment.lower()
    base = "edens"

    # Detect function 25 / vdW potential runs (your "25 -> 0" route)
    if ("van der waals potential" in seg_l) or ("vdw potential" in seg_l) or ("function 25" in seg_l):
        base = "func25"
    # If user truly did vdW molecular surface option, sometimes appears in text
    elif ("van der waals molecular surface" in seg_l) or ("vdw surface" in seg_l):
        base = "vdw"
    # Other possible surfaces (future-proof)
    elif "hirshfeld" in seg_l:
        base = "hirshfeld"
    elif "becke" in seg_l:
        base = "becke"
    elif surface_def and "electron density" not in surface_def.lower():
        base = _norm_simple(surface_def)

    return f"{base}_iso{_fmt_iso(iso)}"


def _norm_simple(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def parse_multiwfn_log(text: str) -> List[Tuple[str, PartCSurfaceMetrics]]:
    """
    Return list of (loaded_wfx_path, metrics) for ALL surface-analysis runs in the log.
    """
    sessions = list(LOAD_RE.finditer(text))
    if not sessions:
        return []

    all_results: List[Tuple[str, PartCSurfaceMetrics]] = []

    for si, sm in enumerate(sessions):
        s_start = sm.start()
        s_end = sessions[si + 1].start() if (si + 1) < len(sessions) else len(text)
        session = text[s_start:s_end]
        wfx_path = sm.group(1).strip()

        # Find all summary blocks inside this session
        sum_marks = list(SUMMARY_RE.finditer(session))
        if not sum_marks:
            continue

        # For each summary block, take its segment from the closest preceding "Quantitative Molecular Surface Analysis"
        # until "Surface analysis finished!"
        for sum_m in sum_marks:
            seg_start = max(0, session.rfind("Quantitative Molecular Surface Analysis", 0, sum_m.start()))
            seg = session[seg_start: session.find("Surface analysis finished!", sum_m.start()) if session.find("Surface analysis finished!", sum_m.start()) != -1 else len(session)]

            # surface def + iso
            surf_def = None
            iso = None
            mline = SURF_LINE_RE.search(seg)
            if mline:
                surf_def = mline.group(1).strip()
                iso = _to_float(mline.group(2))

            # if not found, try broader window
            if surf_def is None:
                mline2 = SURF_LINE_RE.search(session)
                if mline2:
                    surf_def = mline2.group(1).strip()
                    iso = _to_float(mline2.group(2))

            # tag
            tag = _detect_surface_tag(seg, surf_def, iso)

            # parse total surface area in Å^2
            total_A2 = _to_float(
                (re.search(r"Overall surface area:\s*[-\d.+eE]+\s*Bohr\^2\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^2\)", seg) or [None, None]).group(1)
            ) if re.search(r"Overall surface area:", seg) else None

            # volume (optional)
            vol_A3 = None
            mv = re.search(r"Volume:\s*[-\d.+eE]+\s*Bohr\^3\s*\(\s*([-\d.+eE]+|NaN)\s*Angstrom\^3\)", seg)
            if mv:
                vol_A3 = _to_float(mv.group(1))

            # global min/max ESP (optional)
            minESP = None
            maxESP = None
            mmn = re.search(r"Global surface minimum:\s*([-\d.+eE]+|NaN)\s*a\.u\.", seg)
            mmx = re.search(r"Global surface maximum:\s*([-\d.+eE]+|NaN)\s*a\.u\.", seg)
            if mmn: minESP = _to_float(mmn.group(1))
            if mmx: maxESP = _to_float(mmx.group(1))

            # polar/nonpolar (Å^2)
            thr = None
            nonpol = None
            pol = None
            mnp = re.search(r"Nonpolar surface area\s*\(\|ESP\|\s*<=\s*([0-9.]+)\s*kcal/mol\):\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", seg)
            if mnp:
                thr = _to_float(mnp.group(1))
                nonpol = _to_float(mnp.group(2))
            mp = re.search(r"Polar surface area\s*\(\|ESP\|\s*>\s*[0-9.]+\s*kcal/mol\):\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", seg)
            if mp:
                pol = _to_float(mp.group(1))

            # Fallback if those lines not present (older formats)
            if nonpol is None:
                mnp2 = re.search(r"Nonpolar surface area.*?:\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", seg)
                if mnp2: nonpol = _to_float(mnp2.group(1))
            if pol is None:
                mp2 = re.search(r"Polar surface area.*?:\s*([-\d.+eE]+|NaN)\s*Angstrom\^2", seg)
                if mp2: pol = _to_float(mp2.group(1))

            # If total not found, but pol/nonpol exist, infer total
            if total_A2 is None and pol is not None and nonpol is not None:
                total_A2 = pol + nonpol

            metrics = PartCSurfaceMetrics(
                tag=tag,
                surface_def=surf_def,
                iso=iso,
                total_surface_A2=total_A2,
                polar_area_A2=pol,
                nonpolar_area_A2=nonpol,
                polar_threshold_kcal=thr,
                volume_A3=vol_A3,
                minESP_au=minESP,
                maxESP_au=maxESP,
            )

            all_results.append((wfx_path, metrics))

    return all_results


# -------------------------
# mapping loaded paths -> local wfx in calculations/
# -------------------------

def discover_all_wfx() -> List[Path]:
    if not CALCULATIONS_DIR.exists():
        return []
    return list(CALCULATIONS_DIR.rglob("*.wfx"))


def choose_best_wfx_candidate(loaded_path: str, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    hint = _norm_path(loaded_path)
    hint_name = Path(hint).name
    hint_parent = Path(hint).parent.name

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

    # Fuzzy match
    hn = _norm_simple(hint)
    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        pn = _norm_simple(str(p))
        score = 0
        if _norm_simple(p.name) in hn: score += 5
        if _norm_simple(p.parent.name) in hn: score += 2
        if pn in hn: score += 1
        scored.append((score, p))
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1] if scored else None


# -------------------------
# ingest logs -> write JSON
# -------------------------

def surface_json_path(wfx: Path, tag: str) -> Path:
    # e.g. Ala_InterFrag.wfx.surface.func25_iso0.001.json
    return wfx.with_suffix(wfx.suffix + f".surface.{tag}{JSON_SUFFIX}")


def ingest_logs(paths: List[Path]) -> int:
    candidates = discover_all_wfx()
    written = 0

    for logp in paths:
        if not logp.exists() or not logp.is_file():
            continue
        if logp.suffix.lower() not in {".log", ".txt"}:
            continue

        text = logp.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_multiwfn_log(text)
        if not parsed:
            continue

        for loaded_wfx, metrics in parsed:
            target = choose_best_wfx_candidate(loaded_wfx, candidates)
            if target is None:
                continue
            outp = surface_json_path(target, metrics.tag)
            outp.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
            written += 1

    return written


def discover_logs() -> List[Path]:
    patterns = [
        "**/multiwfn*.log",
        "**/multiwfn*.txt",
        "**/*surface*.log",
        "**/*surface*.txt",
    ]
    logs: List[Path] = []
    seen = set()
    for pat in patterns:
        for p in CALCULATIONS_DIR.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                logs.append(p)
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


# -------------------------
# choose best tag per fragment
# -------------------------

DEFAULT_TAG_PRIORITY = [
    "func25_iso0.001",
    "vdw_iso0.001",
    "edens_iso0.001",
]

def best_surface_json_for_fragment(fragment: str, forced_tag: Optional[str] = None) -> Optional[Path]:
    frag_dir = CALCULATIONS_DIR / fragment
    if not frag_dir.is_dir():
        return None

    # all jsons for this fragment
    all_json = list(frag_dir.rglob("*.wfx.surface.*.json"))
    if not all_json:
        return None

    if forced_tag:
        cand = [p for p in all_json if p.name.endswith(f".surface.{forced_tag}.json")]
        if cand:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cand[0]
        return None

    # try priority list
    for tag in DEFAULT_TAG_PRIORITY:
        cand = [p for p in all_json if p.name.endswith(f".surface.{tag}.json")]
        if cand:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cand[0]

    # fallback newest
    all_json.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_json[0]


def load_metrics(p: Path) -> PartCSurfaceMetrics:
    return PartCSurfaceMetrics(**json.loads(p.read_text(encoding="utf-8", errors="ignore")))


# -------------------------
# update my_dataset.csv
# -------------------------

PARTC_COLS = ["QPSA", "QNPSA"]

def update_my_dataset(overwrite: bool = False, tag: Optional[str] = None) -> int:
    if not MY_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing {MY_DATASET_CSV}")

    df = pd.read_csv(MY_DATASET_CSV, index_col=0)
    if "name" not in df.columns or "interactor" not in df.columns:
        raise ValueError("my_dataset.csv must contain 'name' and 'interactor' columns")

    for c in PARTC_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    updated = 0

    for idx, row in df.iterrows():
        name = str(row["name"])
        interactor = str(row["interactor"])
        sugar = name.split("-", 1)[0]

        if not overwrite:
            if pd.notna(row.get("QPSA", pd.NA)) and pd.notna(row.get("QNPSA", pd.NA)):
                print(f"[{idx}] {name}: already computed — skipping.")
                continue

        sj = best_surface_json_for_fragment(sugar, forced_tag=tag)
        ij = best_surface_json_for_fragment(interactor, forced_tag=tag)

        if sj is None:
            print(f"[{idx}] {name}: surface JSON not found for sugar '{sugar}' — skipping.")
            continue
        if ij is None:
            print(f"[{idx}] {name}: surface JSON not found for interactor '{interactor}' — skipping.")
            continue

        sm = load_metrics(sj)
        im = load_metrics(ij)

        if sm.polar_area_A2 is None or sm.nonpolar_area_A2 is None:
            print(f"[{idx}] {name}: missing polar/nonpolar areas for sugar — skipping.")
            continue
        if im.polar_area_A2 is None or im.nonpolar_area_A2 is None:
            print(f"[{idx}] {name}: missing polar/nonpolar areas for interactor — skipping.")
            continue

        QPSA = sm.polar_area_A2 + im.polar_area_A2
        QNPSA = sm.nonpolar_area_A2 + im.nonpolar_area_A2

        df.at[idx, "QPSA"] = QPSA
        df.at[idx, "QNPSA"] = QNPSA

        print(f"[{idx}] {name}: computed Part-C (QPSA/QNPSA) using tag sugar={sm.tag}, interactor={im.tag}")
        updated += 1

    df.to_csv(MY_DATASET_CSV)
    print(f"\nFinished. Updated {updated} row(s). Saved to: {MY_DATASET_CSV}")
    return updated


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", nargs="*", default=[], help="Explicit log file(s) to ingest")
    ap.add_argument("--ingest-logs", action="store_true", help="Auto-discover logs under calculations/")
    ap.add_argument("--update", action="store_true", help="Update my_dataset.csv")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing values")
    ap.add_argument("--tag", default=None, help="Force a specific tag (e.g. func25_iso0.001)")
    args = ap.parse_args()

    # no-args mode: behave like compute_homo_lumo.py
    if len(sys.argv) == 1:
        args.ingest_logs = True
        args.update = True
        args.overwrite = False

    # ingest explicit logs
    if args.ingest:
        n = ingest_logs([Path(p) for p in args.ingest])
        print(f"Ingested explicit logs: wrote {n} JSON file(s).")

    # auto-ingest
    if args.ingest_logs or args.update:
        logs = discover_logs()
        if logs:
            n = ingest_logs(logs)
            print(f"Auto-ingested logs: wrote {n} JSON file(s) from {len(logs)} log(s).")
        else:
            print(f"WARNING: no logs found under {CALCULATIONS_DIR}")

    # update
    if args.update:
        update_my_dataset(overwrite=args.overwrite, tag=args.tag)


if __name__ == "__main__":
    main()