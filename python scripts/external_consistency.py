#!/usr/bin/env python3
"""
Compare all AMS FTAs against an ASEAN FTA baseline using WCopyfind-analogous rules.

Rules implemented:
- Ignore punctuation, outer punctuation, numbers, letter case; skip non-words
  => tokenize to [A-Za-z]+ only, lowercased
- Identify verbatim contiguous matching sequences of >= K words (default K=6)
- Enforce non-overlapping matches (matched words may not be reused)
- Count matched words by full match length (maximally-extended contiguous runs)

Method (WCopyfind-analogous):
1) Tokenize & normalize
2) For each AMS FTA:
   a) Seed candidate matches via hashed K-grams
   b) Verify exact equality of K-gram and extend forward to maximal contiguous match
   c) Select non-overlapping matches (greedy longest-first) on BOTH documents
   d) matched_words = sum(length of chosen matches)
3) Report:
   - matched_words
   - (matched_words / total_words_ams) * 100
   - (matched_words / total_words_asean) * 100
   Rounded to 2 decimals

Outputs:
- CSV with one row per AMS FTA vs ASEAN baseline
- Optional companion CSV listing chosen matches (token index spans)

Usage:
  python3 compare_ams_to_asean_fta.py --asean "ASEAN FTA.txt" --dir "/path/to/ams_ftas" --out results.csv
  python3 compare_ams_to_asean_fta.py --asean "ASEAN FTA.txt" --files "CA-China (2020).txt" "CA-ROK (2021).txt"
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

WORD_RE = re.compile(r"[A-Za-z]+")  # skip numbers and non-words


@dataclass(frozen=True)
class Match:
    length: int
    a_start: int  # start index in AMS tokens
    b_start: int  # start index in ASEAN tokens


def tokenize_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def kgram_digest(tokens: List[str], i: int, k: int) -> bytes:
    """
    64-bit blake2b digest of a K-gram with delimiters.
    Collisions are guarded by exact token verification before extension.
    """
    h = hashlib.blake2b(digest_size=8)
    for t in tokens[i : i + k]:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.digest()


def build_seed_index(tokens: List[str], k: int) -> Dict[bytes, List[int]]:
    idx: Dict[bytes, List[int]] = {}
    limit = len(tokens) - k + 1
    for i in range(limit):
        d = kgram_digest(tokens, i, k)
        idx.setdefault(d, []).append(i)
    return idx


def extend_forward(a: List[str], b: List[str], ai: int, bi: int, k: int) -> int:
    """
    Precondition: a[ai:ai+k] == b[bi:bi+k]
    Extend forward maximally; return length L >= k.
    """
    L = k
    while ai + L < len(a) and bi + L < len(b) and a[ai + L] == b[bi + L]:
        L += 1
    return L


def collect_candidate_matches(ams: List[str], asean: List[str], k: int) -> List[Match]:
    """
    Hash-seed + verify + extend.
    Index the smaller document for memory efficiency.
    """
    if len(ams) <= len(asean):
        small, large = ams, asean
        small_is_ams = True
    else:
        small, large = asean, ams
        small_is_ams = False

    idx = build_seed_index(small, k)
    candidates: List[Match] = []

    limit = len(large) - k + 1
    for j in range(limit):
        d = kgram_digest(large, j, k)
        starts = idx.get(d)
        if not starts:
            continue
        for i in starts:
            if small[i : i + k] != large[j : j + k]:
                continue
            L = extend_forward(small, large, i, j, k)
            if small_is_ams:
                candidates.append(Match(L, a_start=i, b_start=j))
            else:
                # swapped
                candidates.append(Match(L, a_start=j, b_start=i))

    return candidates


def intervals_overlap(sorted_intervals: List[Tuple[int, int]], s: int, e: int) -> bool:
    """
    sorted_intervals sorted by start; check overlap with [s,e)
    """
    pos = bisect.bisect_left(sorted_intervals, (s, e))
    if pos > 0:
        ps, pe = sorted_intervals[pos - 1]
        if ps < e and s < pe:
            return True
    if pos < len(sorted_intervals):
        ns, ne = sorted_intervals[pos]
        if ns < e and s < ne:
            return True
    return False


def add_interval(sorted_intervals: List[Tuple[int, int]], s: int, e: int) -> None:
    bisect.insort(sorted_intervals, (s, e))


def non_overlapping_tiling(ams: List[str], asean: List[str], k: int) -> Tuple[int, List[Match]]:
    """
    Enforce non-overlap on BOTH docs by selecting matches longest-first.
    Deterministic tie-break: length desc, then ams start asc, then asean start asc.
    """
    candidates = collect_candidate_matches(ams, asean, k)
    candidates.sort(key=lambda m: (-m.length, m.a_start, m.b_start))

    used_ams: List[Tuple[int, int]] = []
    used_asean: List[Tuple[int, int]] = []
    chosen: List[Match] = []

    for m in candidates:
        a_s, a_e = m.a_start, m.a_start + m.length
        b_s, b_e = m.b_start, m.b_start + m.length

        if intervals_overlap(used_ams, a_s, a_e):
            continue
        if intervals_overlap(used_asean, b_s, b_e):
            continue

        add_interval(used_ams, a_s, a_e)
        add_interval(used_asean, b_s, b_e)
        chosen.append(m)

    matched_words = sum(m.length for m in chosen)
    return matched_words, chosen


def list_txt_files(directory: str) -> List[str]:
    files = []
    for name in os.listdir(directory):
        if name.lower().endswith(".txt"):
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all AMS FTAs against ASEAN FTA (WCopyfind-analogous).")
    p.add_argument("--asean", required=True, help="Path to ASEAN FTA baseline .txt file.")
    p.add_argument("--dir", default=None, help="Directory containing AMS .txt files (default: current directory).")
    p.add_argument("--files", nargs="*", default=None, help="Explicit AMS .txt files to compare.")
    p.add_argument("--k", type=int, default=6, help="Minimum contiguous match length in words (default: 6).")
    p.add_argument("--out", default="ams_vs_asean_results.csv", help="Output CSV path.")
    p.add_argument(
        "--emit-matches",
        action="store_true",
        help="Also write a companion CSV of the chosen (non-overlapping) matches per AMS file (can be large).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1:
        print("Error: --k must be >= 1", file=sys.stderr)
        return 2

    if not os.path.exists(args.asean):
        print(f"Error: ASEAN baseline file not found: {args.asean}", file=sys.stderr)
        return 2

    # Resolve AMS file list
    if args.files and len(args.files) > 0:
        ams_paths = args.files
    else:
        directory = args.dir if args.dir else os.getcwd()
        ams_paths = list_txt_files(directory)

    # Exclude the ASEAN file if it happens to be in the same dir list
    asean_abs = os.path.abspath(args.asean)
    ams_paths = [p for p in ams_paths if os.path.abspath(p) != asean_abs]

    if len(ams_paths) < 1:
        print("Error: Need at least one AMS .txt file to compare.", file=sys.stderr)
        return 2

    # Tokenize ASEAN baseline once
    asean_tokens = tokenize_file(args.asean)
    total_asean = len(asean_tokens)

    rows = []
    match_rows = []

    for ams_path in ams_paths:
        if not os.path.exists(ams_path):
            print(f"Error: AMS file not found: {ams_path}", file=sys.stderr)
            return 2

        ams_tokens = tokenize_file(ams_path)
        total_ams = len(ams_tokens)

        matched, chosen = non_overlapping_tiling(ams_tokens, asean_tokens, args.k)

        pct_ams = round((matched / total_ams) * 100, 2) if total_ams else 0.0
        pct_asean = round((matched / total_asean) * 100, 2) if total_asean else 0.0

        rows.append([
            os.path.basename(ams_path),
            os.path.basename(args.asean),
            matched,
            total_ams,
            total_asean,
            f"{pct_ams:.2f}",
            f"{pct_asean:.2f}",
        ])

        if args.emit_matches:
            pair_id = f"{os.path.basename(ams_path)}__VS__{os.path.basename(args.asean)}"
            for m in chosen:
                match_rows.append([
                    pair_id,
                    m.length,
                    m.a_start,
                    m.a_start + m.length,
                    m.b_start,
                    m.b_start + m.length,
                ])

        print(f"{os.path.basename(ams_path)} vs {os.path.basename(args.asean)}: matched_words={matched}")

    # Write main CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "AMS FTA",
            "ASEAN baseline",
            "Matched words",
            "Total words AMS",
            "Total words ASEAN",
            "% of AMS",
            "% of ASEAN",
        ])
        w.writerows(rows)

    if args.emit_matches:
        out2 = os.path.splitext(args.out)[0] + "_matches.csv"
        with open(out2, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Pair",
                "Match length (words)",
                "AMS start",
                "AMS end",
                "ASEAN start",
                "ASEAN end",
            ])
            w.writerows(match_rows)

    print(f"Wrote: {args.out}")
    if args.emit_matches:
        print(f"Wrote: {os.path.splitext(args.out)[0] + '_matches.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())