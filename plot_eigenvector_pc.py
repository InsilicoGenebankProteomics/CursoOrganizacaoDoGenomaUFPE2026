#!/usr/bin/env python3
"""Plot a Juicer Tools eigenvector (PC1) as black bars.

All finite values are shown in the same black color, with each bar representing
one genomic bin. The sign of PC1 is arbitrary; the plot does not assign A/B
compartment labels.

Examples:
  python3 plot_eigenvector.py eigenvector_chr1.txt --chromosome chr1 --bin-size 100000
  python3 plot_eigenvector.py eigenvector_chr1.txt --chromosome chr1 --bin-size 100000 \\
      --output chr1_PC1.pdf --png chr1_PC1.png --bedgraph chr1_PC1.bedGraph --no-show
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np


def read_vector(filename):
    try:
        values = np.loadtxt(filename, comments="#", ndmin=1, dtype=float)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read numeric eigenvector values from {filename!r}. "
            "Expected one number (or NaN) per line; check the Juicer Tools output."
        ) from exc
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{filename!r} contains no values.")
    return values


def write_bedgraph(path, chromosome, values, bin_size, start, chrom_length=None):
    with open(path, "w", encoding="utf-8") as handle:
        for i, value in enumerate(values):
            bin_start = start + i * bin_size
            bin_end = bin_start + bin_size
            if chrom_length is not None:
                bin_end = min(bin_end, chrom_length)
            if bin_end <= bin_start:
                break
            val = f"{value:.8g}" if np.isfinite(value) else "nan"
            handle.write(f"{chromosome}\t{bin_start}\t{bin_end}\t{val}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a Juicer Tools eigenvector (PC1) using black bars."
    )
    parser.add_argument("input", help="One-column eigenvector output file.")
    parser.add_argument("--chromosome", default="chr1", help="Chromosome ID (default: chr1).")
    parser.add_argument("--component", type=int, default=1,
                        help="Principal component number represented in the input (default: 1).")
    parser.add_argument("--bin-size", type=int, default=100000,
                        help="Resolution in bp used for eigenvector (default: 100000).")
    parser.add_argument("--start", type=int, default=0, help="First bin start in bp (default: 0).")
    parser.add_argument("--chrom-length", type=int, default=None,
                        help="Optional chromosome length in bp, to trim the last bedGraph bin.")
    parser.add_argument("--output", default="eigenvector.pdf", help="PDF output filename.")
    parser.add_argument("--png", default="eigenvector.png", help="PNG output filename.")
    parser.add_argument("--bedgraph", default=None,
                        help="Optional bedGraph output for a genome-browser track.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save files without opening a plot window.")
    args = parser.parse_args()

    if args.component < 1:
        parser.error("--component must be a positive integer (1=PC1, 2=PC2, etc.).")
    if args.bin_size <= 0:
        parser.error("--bin-size must be positive.")
    if args.start < 0:
        parser.error("--start must be non-negative.")
    if args.chrom_length is not None and args.chrom_length <= args.start:
        parser.error("--chrom-length must exceed --start.")

    try:
        vals = read_vector(args.input)
    except ValueError as exc:
        parser.error(str(exc))

    finite = np.isfinite(vals)
    if not finite.any():
        parser.error("No finite eigenvector values were found.")

    centers_mb = (args.start + (np.arange(len(vals)) + 0.5) * args.bin_size) / 1e6
    width_mb = args.bin_size / 1e6
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(centers_mb[finite], vals[finite], width=width_mb, color="black",
           edgecolor="black", linewidth=0, align="center")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{args.chromosome} position (Mb)")
    ax.set_ylabel(f"Eigenvector (PC{args.component})")
    ax.set_title(f"Hi-C eigenvector (PC{args.component}): {args.chromosome}")
    ax.margins(x=0.005)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.png, dpi=300, bbox_inches="tight")

    if args.bedgraph:
        write_bedgraph(args.bedgraph, args.chromosome, vals, args.bin_size,
                       args.start, args.chrom_length)

    print(f"Input bins: {len(vals)}")
    print(f"Finite values: {int(finite.sum())}")
    print(f"PDF: {args.output}")
    print(f"PNG: {args.png}")
    if args.bedgraph:
        print(f"bedGraph: {args.bedgraph}")
    print(f"All bars use the same color; PC{args.component} sign is arbitrary and is not labeled as A/B.")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
