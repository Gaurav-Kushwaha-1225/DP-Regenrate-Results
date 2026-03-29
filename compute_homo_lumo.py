#!/usr/bin/env python3
"""
compute_homo_lumo.py

Computes HOMO-LUMO MO-gap descriptors from ORCA .out fragment files
and updates my_dataset.csv with newly available data.

Expected file layout:
    calculations/{FragmentName}/{FragmentName}_InterFrag.out

Sugar name  : parsed from the 'name' column  (e.g. "GlcNAc-Ala" -> sugar="GlcNAc")
Interactor  : taken directly from the 'interactor' column (e.g. "Ala")

Descriptor formula (energies in Hartree):
    {X}s-{Y}i = E(sugar orbital X) - E(interactor orbital Y)

MO index rules (offset from HOMO index):
    HOMO     ->  0      HOMO-K  -> -K
    LUMO     -> +1      LUMO+M  -> +(1+M)

Usage:
    python compute_homo_lumo.py

Run this script each time you add a new interactor .out file.
Rows that already have values are skipped automatically.
"""

import os
import re
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALCULATIONS_DIR = os.path.join(BASE_DIR, "calculations")
MY_DATASET_CSV = os.path.join(BASE_DIR, "my_dataset.csv")


# ---------------------------------------------------------------------------
# Parsing ORCA .out orbital energies
# ---------------------------------------------------------------------------

def parse_orbital_energies(out_file: str):
    """
    Extract the last ORBITAL ENERGIES table from an ORCA .out file.

    The last table corresponds to the final converged SCF step.
    Returns (occupancies, energies_Eh) as parallel lists indexed by
    orbital number (NO column).
    """
    best_occs = []
    best_energies = []

    with open(out_file, "r") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() == "ORBITAL ENERGIES":
            # Advance past separator "---" and blank line to reach header
            j = i + 1
            while j < len(lines) and "NO" not in lines[j]:
                j += 1
            j += 1  # skip the "NO  OCC  E(Eh)  E(eV)" header line

            curr_occs = []
            curr_energies = []
            while j < len(lines):
                row = lines[j].strip()
                if not row:
                    break  # blank line terminates the table
                parts = row.split()
                if len(parts) == 4:
                    try:
                        curr_occs.append(float(parts[1]))    # OCC
                        curr_energies.append(float(parts[2]))  # E(Eh)
                    except ValueError:
                        break
                j += 1

            if curr_occs:
                best_occs = curr_occs
                best_energies = curr_energies
            i = j
        else:
            i += 1

    if not best_occs:
        raise ValueError(f"No ORBITAL ENERGIES table found in: {out_file}")

    return best_occs, best_energies


def get_homo_index(occupancies: list) -> int:
    """
    Return the HOMO index as the last occupied orbital.

    Works for:
    - closed-shell tables (occupied OCC ~ 2.0000)
    - open-shell/spin-resolved tables (occupied OCC ~ 1.0000)
    - any fractional-occupancy case where OCC > 0
    """
    homo_idx = -1
    for i, occ in enumerate(occupancies):
        if occ > 1e-8:
            homo_idx = i
    if homo_idx == -1:
        raise ValueError("No occupied orbital (OCC>0) found in table.")
    return homo_idx


# ---------------------------------------------------------------------------
# Column name parsing
# ---------------------------------------------------------------------------

def parse_mo_offset(mo_name: str) -> int:
    """
    Convert an MO name to an integer index offset from HOMO.

        HOMO     ->  0
        HOMO-K   -> -K
        LUMO     -> +1
        LUMO+M   -> +(1 + M)
    """
    if mo_name == "HOMO":
        return 0
    m = re.match(r"^HOMO-(\d+)$", mo_name)
    if m:
        return -int(m.group(1))
    if mo_name == "LUMO":
        return 1
    m = re.match(r"^LUMO\+(\d+)$", mo_name)
    if m:
        return 1 + int(m.group(1))
    raise ValueError(f"Unrecognised MO name: '{mo_name}'")


# Column names use uppercase MO labels followed by lowercase 's' (sugar)
# or 'i' (interactor), e.g. "HOMO-5s-LUMO+1i".
# [A-Z0-9+\-]+ cannot match the lowercase 's'/'i' separators, so the
# greedy match naturally stops at the right boundary.
_COL_RE = re.compile(r"^((?:HOMO|LUMO)(?:[+-]\d+)?)s-((?:HOMO|LUMO)(?:[+-]\d+)?)i$")

def parse_column_name(col: str):
    """
    Parse a descriptor column name like "HOMO-5s-LUMO+1i" into
    (sugar_offset, interactor_offset) as integers.
    Returns None if the column is not an MO-gap descriptor.
    """
    m = _COL_RE.match(col)
    if not m:
        return None
    try:
        return parse_mo_offset(m.group(1)), parse_mo_offset(m.group(2))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def get_orbital_energy(energies: list, homo_idx: int, offset: int, label: str) -> float:
    idx = homo_idx + offset
    if idx < 0 or idx >= len(energies):
        raise IndexError(
            f"Orbital '{label}' maps to index {idx}, "
            f"but table only has indices 0–{len(energies) - 1}."
        )
    return energies[idx]


def compute_descriptors(sugar_out: str, inter_out: str, mo_columns: list) -> dict:
    """
    Compute all MO-gap descriptor values for a sugar/interactor pair.
    Returns {column_name: float_value}.  Missing orbitals yield NaN.
    """
    sugar_occs, sugar_energies = parse_orbital_energies(sugar_out)
    inter_occs, inter_energies = parse_orbital_energies(inter_out)

    sugar_homo = get_homo_index(sugar_occs)
    inter_homo = get_homo_index(inter_occs)

    print(f"    Sugar  HOMO index: {sugar_homo}  E={sugar_energies[sugar_homo]:.6f} Eh")
    print(f"    Inter. HOMO index: {inter_homo}  E={inter_energies[inter_homo]:.6f} Eh")

    results = {}
    for col in mo_columns:
        parsed = parse_column_name(col)
        if parsed is None:
            continue
        s_off, i_off = parsed
        try:
            e_s = get_orbital_energy(sugar_energies, sugar_homo, s_off, f"sugar+{s_off}")
            e_i = get_orbital_energy(inter_energies, inter_homo, i_off, f"inter+{i_off}")
            results[col] = e_s - e_i
        except IndexError as exc:
            results[col] = float("nan")
            print(f"    WARNING {col}: {exc}")

    return results


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_out_file(fragment_name: str) -> str | None:
    """
    Return the path to {fragment_name}_InterFrag.out under
    calculations/{fragment_name}/, or None if it does not exist.
    """
    path = os.path.join(
        CALCULATIONS_DIR, fragment_name, f"{fragment_name}_InterFrag.out"
    )
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(MY_DATASET_CSV):
        raise FileNotFoundError(f"Dataset not found: {MY_DATASET_CSV}")

    df = pd.read_csv(MY_DATASET_CSV, index_col=0)

    # Identify all MO-gap descriptor columns present in the CSV
    mo_columns = [c for c in df.columns if parse_column_name(c) is not None]
    print(f"MO-gap descriptor columns detected: {len(mo_columns)}")

    # Keep only the identity columns and MO-gap descriptor columns;
    # drop everything else (ddGglyc … MaxESP and any other descriptors).
    identity_cols = [c for c in ("name", "interactor") if c in df.columns]
    keep_cols = identity_cols + mo_columns
    df = df[keep_cols]

    if not mo_columns:
        print("No MO-gap columns found in CSV. Nothing to compute.")
        return

    first_mo_col = mo_columns[0]
    updated_rows = 0

    for idx, row in df.iterrows():
        name = str(row["name"])           # e.g. "GlcNAc-Ala"
        interactor = str(row["interactor"])  # e.g. "Ala"

        # Sugar is the first part of the complex name, e.g. "GlcNAc"
        sugar_name = name.split("-", 1)[0]

        # Skip rows that already have values computed
        val = row[first_mo_col]
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            print(f"[{idx}] {name}: already computed — skipping.")
            continue

        # Locate the .out files
        sugar_out = find_out_file(sugar_name)
        inter_out = find_out_file(interactor)

        if sugar_out is None:
            print(f"[{idx}] {name}: .out file not found for sugar '{sugar_name}' — skipping.")
            continue
        if inter_out is None:
            print(f"[{idx}] {name}: .out file not found for interactor '{interactor}' — skipping.")
            continue

        print(f"[{idx}] {name}: computing {len(mo_columns)} descriptors...")
        print(f"    Sugar file : {os.path.relpath(sugar_out, BASE_DIR)}")
        print(f"    Inter file : {os.path.relpath(inter_out, BASE_DIR)}")

        try:
            descriptors = compute_descriptors(sugar_out, inter_out, mo_columns)
        except Exception as exc:
            print(f"    ERROR: {exc} — skipping row.")
            continue

        for col, val in descriptors.items():
            df.at[idx, col] = val

        updated_rows += 1
        nan_count = sum(1 for v in descriptors.values() if pd.isna(v))
        print(f"    Done ({len(descriptors) - nan_count} values, {nan_count} NaN).")

    df.to_csv(MY_DATASET_CSV)
    print(f"\nFinished. Updated {updated_rows} row(s). Saved to: {MY_DATASET_CSV}")


if __name__ == "__main__":
    main()
