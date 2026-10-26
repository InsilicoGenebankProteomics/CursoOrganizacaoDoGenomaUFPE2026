#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Hi-C BAM summary
#
# Usage:
#   bash hic_bam_summary.sh merged_dedup.bam sample_name
#
# Output:
#   sample_name_hic_summary.csv
#
# The calculations follow the logic of the older Hi-C R script:
#   - duplicate pairs
#   - non-duplicate MAPQ >= 10 pairs
#   - pairs >= 1 kb
#   - pairs >= 1 Mb
#   - interchromosomal pairs
#
# Important:
#   merged_dedup.bam does NOT contain the original FASTQ count.
#   "bam_pairs" is the number of primary paired reads represented
#   in this BAM.
# ============================================================

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 merged_dedup.bam [sample_name]" >&2
    exit 1
fi

BAM="$1"

if [[ ! -f "$BAM" ]]; then
    echo "ERROR: BAM file not found: $BAM" >&2
    exit 1
fi

SAMPLE="${2:-$(basename "$BAM" .bam)}"
OUT="${SAMPLE}_hic_summary.csv"
TMP="${SAMPLE}_hic_pairs.tmp"

# Prefer GNU awk when available, because Juicer workflows commonly
# use awk features that are more reliable with gawk on macOS.
if command -v gawk >/dev/null 2>&1; then
    AWK="gawk"
elif command -v awk >/dev/null 2>&1; then
    AWK="awk"
else
    echo "ERROR: awk/gawk not found." >&2
    exit 1
fi

if ! command -v samtools >/dev/null 2>&1; then
    echo "ERROR: samtools not found in PATH." >&2
    exit 1
fi

echo "BAM:    $BAM"
echo "Sample: $SAMPLE"
echo
echo "Counting paired primary alignments..."

# ------------------------------------------------------------
# Step 1: keep primary paired alignments and collate by QNAME
#
# -f 1      = read is part of a paired-end template
# -F 0x900  = remove secondary (0x100) and supplementary (0x800)
#
# We KEEP duplicate reads here because we want to count them.
# samtools collate groups the two mates by read name without
# requiring a full coordinate -> name sort.
# ------------------------------------------------------------

samtools view -u -f 1 -F 0x900 "$BAM" \
    | samtools collate -O -u - \
    | samtools view - \
    | "$AWK" -v OFS='\t' '
BEGIN {
    # Pair-level counters
    bam_pairs = 0
    duplicate_pairs = 0
    both_mapped_pairs = 0
    singleton_pairs = 0

    non_dup_q10 = 0
    pairs_ge_1kb = 0
    pairs_ge_1mb = 0
    interchromosomal = 0
}

function finish_pair(    mapped1, mapped2, dup, samechr, dist, good) {

    # We only expect two primary records per paired QNAME.
    # Ignore anything incomplete.
    if (n != 2)
        return

    bam_pairs++

    # BAM duplicate flag = 0x400 = decimal 1024.
    dup = (and(flag1, 1024) || and(flag2, 1024))

    if (dup)
        duplicate_pairs++

    mapped1 = !(and(flag1, 4) != 0)
    mapped2 = !(and(flag2, 4) != 0)

    if (mapped1 && mapped2)
        both_mapped_pairs++
    else if (mapped1 || mapped2)
        singleton_pairs++

    # Old R workflow used samtools -q 10 and excluded duplicates,
    # unmapped, secondary and supplementary reads.
    #
    # At pair level we require BOTH mates to be mapped and MAPQ >= 10.
    good = (mapped1 && mapped2 && mapq1 >= 10 && mapq2 >= 10 && !dup)

    if (!good)
        return

    non_dup_q10++

    samechr = (chr1 != "*" && chr2 != "*" && chr1 == chr2)

    # The old R code defined:
    #   not_pe = chr1 != chr2 | abs(start1-start2) >= 1 kb
    #
    # Thus interchromosomal pairs are also included in the >=1 kb group.
    if (!samechr) {
        pairs_ge_1kb++
        interchromosomal++
        return
    }

    dist = start1 - start2
    if (dist < 0)
        dist = -dist

    if (dist >= 1000)
        pairs_ge_1kb++

    if (dist >= 1000000)
        pairs_ge_1mb++
}

{
    qname = $1
    flag  = $2
    chr   = $3
    pos   = $4
    mapq  = $5

    # QNAME changes -> finish the previous pair.
    if (current_qname != "" && qname != current_qname) {
        finish_pair()

        n = 0
    }

    current_qname = qname
    n++

    # Keep the first and second primary records separately.
    if (n == 1) {
        flag1  = flag
        chr1   = chr
        start1 = pos
        mapq1  = mapq
    }
    else if (n == 2) {
        flag2  = flag
        chr2   = chr
        start2 = pos
        mapq2  = mapq
    }
}

END {
    # Finish the final QNAME.
    finish_pair()

    # Fractions
    duplicate_fraction = (bam_pairs > 0) ?
                         duplicate_pairs / bam_pairs : 0

    non_dup_q10_fraction = (bam_pairs > 0) ?
                           non_dup_q10 / bam_pairs : 0

    fraction_ge_1kb = (non_dup_q10 > 0) ?
                      pairs_ge_1kb / non_dup_q10 : 0

    # Same denominator as old R code:
    # d_1Mb / d_1kb
    fraction_ge_1mb = (pairs_ge_1kb > 0) ?
                      pairs_ge_1mb / pairs_ge_1kb : 0

    # Same denominator as old R code:
    # diff_chr / d_1kb
    fraction_interchromosomal = (pairs_ge_1kb > 0) ?
                                 interchromosomal / pairs_ge_1kb : 0

    singleton_fraction = (bam_pairs > 0) ?
                         singleton_pairs / bam_pairs : 0

    # --------------------------------------------------------
    # CSV output
    # --------------------------------------------------------

    print "sample,bam_pairs,duplicate_pairs,duplicate_fraction," \
          "both_mapped_pairs,singleton_pairs,singleton_fraction," \
          "non_dup_q10,p_non_dup_q10,pairs_ge_1kb,p_ge_1kb," \
          "pairs_ge_1mb,p_ge_1mb_among_ge_1kb," \
          "interchromosomal,p_interchromosomal_among_ge_1kb"

    print SAMPLE "," \
          bam_pairs "," \
          duplicate_pairs "," \
          duplicate_fraction "," \
          both_mapped_pairs "," \
          singleton_pairs "," \
          singleton_fraction "," \
          non_dup_q10 "," \
          non_dup_q10_fraction "," \
          pairs_ge_1kb "," \
          fraction_ge_1kb "," \
          pairs_ge_1mb "," \
          fraction_ge_1mb "," \
          interchromosomal "," \
          fraction_interchromosomal
}
' > "$TMP"

# The temporary file contains the single CSV row.
# Add it directly to the final CSV.
mv "$TMP" "$OUT"

# ------------------------------------------------------------
# Human-readable summary
# ------------------------------------------------------------

"$AWK" -F',' '
NR == 1 {
    for (i = 1; i <= NF; i++)
        header[i] = $i
    next
}
NR == 2 {
    for (i = 1; i <= NF; i++)
        value[i] = $i
}
END {
    printf "\n============================================\n"
    printf "Hi-C BAM summary: %s\n", value[1]
    printf "============================================\n"
    printf "Primary paired BAM pairs:      %12d\n", value[2]
    printf "Duplicate pairs:               %12d  (%.2f%%)\n",
           value[3], 100*value[4]
    printf "Both mates mapped:              %12d\n", value[5]
    printf "Singleton pairs:                %12d  (%.2f%%)\n",
           value[6], 100*value[7]

    printf "\nOld-R-style Hi-C summary\n"
    printf "--------------------------------------------\n"
    printf "Non-duplicate MAPQ>=10:         %12d  (%.2f%%)\n",
           value[8], 100*value[9]
    printf "Pairs >= 1 kb:                  %12d  (%.2f%%)\n",
           value[10], 100*value[11]
    printf "Pairs >= 1 Mb:                  %12d  (%.2f%% of >=1 kb)\n",
           value[12], 100*value[13]
    printf "Interchromosomal pairs:         %12d  (%.2f%% of >=1 kb)\n",
           value[14], 100*value[15]

    printf "\nCSV: %s\n\n", "'"$OUT"'"
}
' "$OUT"
