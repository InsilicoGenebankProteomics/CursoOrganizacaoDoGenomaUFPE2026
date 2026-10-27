#!/usr/bin/env python3
"""Summarize HiCCUPS loop and Arrowhead domain BEDPE outputs.

Loop size is defined as the absolute genomic distance between the midpoints
of the two anchors for cis (same-chromosome) loops. Inter-chromosomal loops,
if present, are counted but excluded from distance statistics.
Domain/TAD size is defined as end1 - start1 from the first BEDPE interval;
Arrowhead diagonal domains ordinarily have matching first and second intervals.
"""

import argparse
import statistics
import sys
from pathlib import Path


def read_bedpe(path):
    """Yield (chrom1, start1, end1, chrom2, start2, end2) from BEDPE rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                start1, end1, start2, end2 = map(int, (fields[1], fields[2], fields[4], fields[5]))
            except ValueError:
                # Skips header rows such as: chr1 x1 x2 chr2 y1 y2
                continue
            if start1 < 0 or start2 < 0 or end1 < start1 or end2 < start2:
                print(f"Warning: skipping invalid coordinates at {path}:{line_no}", file=sys.stderr)
                continue
            yield fields[0], start1, end1, fields[3], start2, end2


def report_sizes(label, sizes):
    if not sizes:
        print("  Mean size: not available (no eligible records)")
        print("  Median size: not available (no eligible records)")
        return
    mean_bp = statistics.mean(sizes)
    median_bp = statistics.median(sizes)
    print(f"  Mean:   {mean_bp:,.0f} bp ({mean_bp / 1e6:.4f} Mb)")
    print(f"  Median: {median_bp:,.0f} bp ({median_bp / 1e6:.4f} Mb)")
    print(f"  Min–max: {min(sizes):,}–{max(sizes):,} bp")


def summarize_loops(path):
    rows = list(read_bedpe(path))
    cis_distances = []
    trans_count = 0
    invalid_distance = 0
    for chrom1, start1, end1, chrom2, start2, end2 in rows:
        if chrom1 != chrom2:
            trans_count += 1
            continue
        mid1 = (start1 + end1) / 2
        mid2 = (start2 + end2) / 2
        distance = abs(mid2 - mid1)
        if distance == 0:
            invalid_distance += 1
        cis_distances.append(distance)

    print(f"LOOPS — {path}")
    print(f"  Total loops (valid BEDPE records): {len(rows):,}")
    print(f"  Intra-chromosomal loops:           {len(rows) - trans_count:,}")
    if trans_count:
        print(f"  Inter-chromosomal loops:           {trans_count:,} (excluded from size statistics)")
    if invalid_distance:
        print(f"  Note: {invalid_distance:,} loop(s) have zero midpoint separation")
    print("  Size definition: distance between the midpoints of the two anchors (same chromosome).")
    report_sizes("Loop size", cis_distances)


def summarize_domains(path):
    rows = list(read_bedpe(path))
    sizes = [end1 - start1 for chrom1, start1, end1, chrom2, start2, end2 in rows
             if end1 >= start1]
    mismatched = sum(1 for c1, s1, e1, c2, s2, e2 in rows
                     if (c1 != c2) or (s1 != s2) or (e1 != e2))
    print(f"DOMAINS / TAD-LIKE BLOCKS — {path}")
    print(f"  Total domains (valid BEDPE records): {len(rows):,}")
    print("  Size definition: end1 - start1 (domain width along the diagonal).")
    if mismatched:
        print(f"  Note: {mismatched:,} row(s) have non-identical first/second intervals or chromosomes;")
        print("        confirm that this file contains Arrowhead diagonal domains before interpreting size.")
    report_sizes("Domain size", sizes)


def main():
    parser = argparse.ArgumentParser(
        description="Count HiCCUPS loops and Arrowhead domains and summarize their sizes."
    )
    parser.add_argument("--loops", help="Path to HiCCUPS merged_loops.bedpe")
    parser.add_argument("--tads", help="Path to Arrowhead *blocks.bedpe")
    args = parser.parse_args()
    if not args.loops and not args.tads:
        parser.error("provide --loops and/or --tads")
    try:
        if args.loops:
            summarize_loops(args.loops)
        if args.tads:
            if args.loops:
                print()
            summarize_domains(args.tads)
    except (OSError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
