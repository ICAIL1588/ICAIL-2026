#!/usr/bin/env python3
"""
WCopyfind-analogous pairwise comparison for treaty/FTA .txt files.

Rules implemented (per user spec):
- Ignore punctuation, outer punctuation, numbers, letter case; skip non-words
  => tokenize to [A-Za-z]+ words only, lowercased
- Identify verbatim contiguous matching sequences of >= K words (default K=6)
- Enforce non-overlapping matches (matched words may not be reused)
- Count matched words by full match length (maximally-extended contiguous runs)

Method (WCopyfind-analogous):
1) Tokenize & normalize each file
2) For each pair:
   a) Seed matches using hashed K-grams (fingerprinting)
   b) Verify equality of K-gram, then extend forward to maximal contiguous match
   c) Select non-overlapping matches via greedy longest-first tiling
   d) Compute matched_words, totals, and directional percentages

Output:
- CSV with one row per unique pair

Usage examples:
  python3 wcopy_pairwise.py --dir "/path/to/txts" --out results.csv
  python3 wcopy_pairwise.py --files "A.txt" "B.txt" "C.txt"

Notes:
- This is exact on word tokens (no stemming, no synonyms), as requested.
- Large files can be slow; the script indexes the smaller file to reduce memory.
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
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

WORD_RE = re.compile(r"[A-Za-z]+")  # skip numbers and non-words


@dataclass(frozen=True)
class Match:
    length: int
    a_start: int
    b_start: int


def tokenize_file(path: str) -> List[str]:
    """
    Tokenization per spec:
    - ignore punctuation/numbers/case
    - skip non-words
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def blake2b_64(data: bytes) -> bytes:
    # 64-bit digest; collisions extremely unlikely; we always verify tokens anyway.
    return hashlib.blake2b(data, digest_size=8).digest()


def kgram_digest(tokens: List[str], i: int, k: int) -> bytes:
    """
    Hash K contiguous tokens into a fixed-size digest.
    Use a delimiter to avoid ambiguity (e.g., ["ab","c"] vs ["a","bc"]).
    """
    h = hashlib.blake2b(digest_size=8)
    for t in tokens[i : i + k]:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.digest()


def build_seed_index(tokens: List[str], k: int) -> Dict[bytes, List[int]]:
    """
    Map hash(K-gram) -> list of starting positions in tokens.
    """
    idx: Dict[bytes, List[int]] = {}
    limit = len(tokens) - k + 1
    for i in range(limit):
        d = kgram_digest(tokens, i, k)
        idx.setdefault(d, []).append(i)
    return idx


def extend_match(a: List[str], b: List[str], ai: int, bi: int, k: int) -> int:
    """
    Given ai/bi where a[ai:ai+k] == b[bi:bi+k], extend forward maximally.
    Return full match length L >= k.
    """
    L = k
    # Extend forward (contiguous)
    while ai + L < len(a) and bi + L < len(b) and a[ai + L] == b[bi + L]:
        L += 1
    return L


def collect_candidate_matches(a: List[str], b: List[str], k: int) -> List[Match]:
    """
    Fingerprint + verify + extend:
    - Build index for smaller sequence to reduce memory
    - Scan K-grams in other sequence, look up candidates, verify, extend
    """
    # Index smaller side to reduce memory footprint
    if len(a) <= len(b):
        small, large = a, b
        small_is_a = True
    else:
        small, large = b, a
        small_is_a = False

    idx = build_seed_index(small, k)

    candidates: List[Match] = []
    limit = len(large) - k + 1
    for j in range(limit):
        d = kgram_digest(large, j, k)
        starts = idx.get(d)
        if not starts:
            continue
        # Verify seed equality and extend
        for i in starts:
            if small[i : i + k] != large[j : j + k]:
                continue
            L = extend_match(small, large, i, j, k)
            if small_is_a:
                candidates.append(Match(L, a_start=i, b_start=j))
            else:
                # swapped: small is b, large is a
                candidates.append(Match(L, a_start=j, b_start=i))

    return candidates


def intervals_overlap(sorted_intervals: List[Tuple[int, int]], s: int, e: int) -> bool:
    """
    Check if [s,e) overlaps any existing interval in a list sorted by start.
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


def non_overlapping_tiling(a: List[str], b: List[str], k: int) -> Tuple[int, List[Match]]:
    """
    Enforce "matched words may not be reused" by selecting non-overlapping matches.
    Greedy strategy: choose longest matches first, then skip overlaps.
    (This is a standard, WCopyfind-like post-processing step for non-overlap constraints.)
    """
    candidates = collect_candidate_matches(a, b, k)

    # Deterministic tie-breakers:
    # - length desc, then a_start asc, then b_start asc
    candidates.sort(key=lambda m: (-m.length, m.a_start, m.b_start))

    usedA: List[Tuple[int, int]] = []
    usedB: List[Tuple[int, int]] = []
    chosen: List[Match] = []

    for m in candidates:
        a_s, a_e = m.a_start, m.a_start + m.length
        b_s, b_e = m.b_start, m.b_start + m.length

        if intervals_overlap(usedA, a_s, a_e):
            continue
        if intervals_overlap(usedB, b_s, b_e):
            continue

        add_interval(usedA, a_s, a_e)
        add_interval(usedB, b_s, b_e)
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
    p = argparse.ArgumentParser(description="WCopyfind-analogous pairwise treaty comparison (>=K-word contiguous matches).")
    p.add_argument("--dir", type=str, default=None, help="Directory containing .txt files (default: current directory).")
    p.add_argument("--files", nargs="*", default=None, help="Explicit list of .txt files to compare.")
    p.add_argument("--k", type=int, default=6, help="Minimum contiguous matching phrase length in words (default: 6).")
    p.add_argument("--out", type=str, default="pairwise_results.csv", help="Output CSV path (default: pairwise_results.csv).")
    p.add_argument("--emit-matches", action="store_true",
                   help="Also write a companion CSV listing the chosen (non-overlapping) matches per pair (can be large).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.k < 1:
        print("Error: --k must be >= 1", file=sys.stderr)
        return 2

    # Resolve file list
    if args.files and len(args.files) > 0:
        paths = args.files
    else:
        directory = args.dir if args.dir else os.getcwd()
        paths = list_txt_files(directory)

    if len(paths) < 2:
        print("Error: Need at least two .txt files to compare.", file=sys.stderr)
        return 2

    # Tokenize all files once
    token_map: Dict[str, List[str]] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 2
        toks = tokenize_file(path)
        token_map[path] = toks
        if len(toks) < args.k:
            # Still compare; it will just have 0 matches
            pass

    # Pairwise comparisons
    rows = []
    match_rows = []  # optional detailed output

    for p1, p2 in combinations(paths, 2):
        a = token_map[p1]
        b = token_map[p2]
        matched, chosen = non_overlapping_tiling(a, b, args.k)

        total_a = len(a)
        total_b = len(b)

        pct_a = round((matched / total_a) * 100, 2) if total_a else 0.0
        pct_b = round((matched / total_b) * 100, 2) if total_b else 0.0

        rows.append([
            os.path.basename(p1),
            os.path.basename(p2),
            matched,
            total_a,
            total_b,
            f"{pct_a:.2f}",
            f"{pct_b:.2f}",
        ])

        if args.emit_matches:
            pair_id = f"{os.path.basename(p1)}__VS__{os.path.basename(p2)}"
            for m in chosen:
                match_rows.append([
                    pair_id,
                    m.length,
                    m.a_start,
                    m.a_start + m.length,
                    m.b_start,
                    m.b_start + m.length,
                ])

    # Write main CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "FTA 1",
            "FTA 2",
            "Matched words",
            "Total words FTA 1",
            "Total words FTA 2",
            "% of FTA 1",
            "% of FTA 2",
        ])
        w.writerows(rows)

    if args.emit_matches:
        out2 = os.path.splitext(args.out)[0] + "_matches.csv"
        with open(out2, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Pair",
                "Match length (words)",
                "FTA1 start",
                "FTA1 end",
                "FTA2 start",
                "FTA2 end",
            ])
            w.writerows(match_rows)

    print(f"Wrote: {args.out}")
    if args.emit_matches:
        print(f"Wrote: {os.path.splitext(args.out)[0] + '_matches.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())