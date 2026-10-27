#!/usr/bin/env python3
"""Calculate any principal component from a dense Juicer Tools Pearson matrix.

Input: the whitespace-delimited square text matrix created by Juicer Tools:
  java -jar juicer_tools.jar pearsons ... chromosome BP binsize pearsons.txt

Output: one value per original bin, in original order. Bins excluded by Juicer
(or otherwise having undefined correlations) are retained as NaN so the vector
stays aligned with the chromosome bins.

Example:
  python3 calculate_pc.py pearsons_chr1_100kb.txt --pc 2 \
      --output chr1_PC2_100kb.txt --eigenvalues chr1_eigenvalues.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_dense_matrix(path: Path) -> np.ndarray:
    """Read a dense numeric text matrix and validate its shape."""
    try:
        matrix = np.loadtxt(path, dtype=float, ndmin=2)
    except Exception as exc:
        raise ValueError(
            f"Could not read '{path}' as a numeric text matrix. Use the text output "
            "from Juicer Tools 'pearsons' (for example, name it pearsons.txt), "
            "not a binary .bin file. The file should contain one matrix row per line."
        ) from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Expected a square matrix, but the file has shape {matrix.shape}. "
            "Check that the entire Pearson matrix was exported."
        )
    if matrix.shape[0] < 2:
        raise ValueError("At least two bins are required to calculate principal components.")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate PC1, PC2, PC3, or another numbered principal component "
            "from a dense Pearson correlation matrix exported by Juicer Tools."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "matrix",
        help="Dense numeric text matrix produced by 'juicer_tools pearsons ... pearsons.txt'.",
    )
    parser.add_argument(
        "--pc", type=int, default=1,
        help="Component rank to extract, counting from 1 (1=PC1, 2=PC2, etc.).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="One-column output vector; if omitted, defaults to PC<N>.txt.",
    )
    parser.add_argument(
        "--eigenvalues", default=None,
        help="Optional TSV containing all eigenvalues sorted from largest to smallest and their variance fractions.",
    )
    parser.add_argument(
        "--keep-sign", action="store_true",
        help="Keep the eigensolver's sign for the selected eigenvector. By default, orient it so its largest absolute value is positive.",
    )
    args = parser.parse_args()

    if args.pc < 1:
        parser.error("--pc must be a positive integer: 1=PC1, 2=PC2, and so on.")

    matrix_path = Path(args.matrix)
    output_path = Path(args.output) if args.output else Path(f"PC{args.pc}.txt")

    try:
        matrix = read_dense_matrix(matrix_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    n_original = matrix.shape[0]

    # Juicer Pearson exports use NaN on the diagonal for bins it excludes.
    # Keep those positions in the final vector as NaN, preserving bin order.
    valid = np.isfinite(np.diag(matrix))
    n_valid = int(valid.sum())
    if n_valid < args.pc:
        print(
            f"Error: only {n_valid} bins have a finite diagonal, so PC{args.pc} cannot be calculated.",
            file=sys.stderr,
        )
        return 2

    sub = matrix[np.ix_(valid, valid)]
    if not np.isfinite(sub).all():
        bad = int(sub.size - np.isfinite(sub).sum())
        print(
            f"Error: found {bad} non-finite value(s) inside the submatrix of bins with a finite diagonal. "
            "The script will not silently impute missing correlations. Check the Pearson matrix; "
            "if a few bins are malformed, remove them consistently from rows and columns first.",
            file=sys.stderr,
        )
        return 2

    # A Pearson matrix should be symmetric. Report large asymmetry, then symmetrize
    # tiny floating-point differences to make eigh's symmetric-matrix assumption explicit.
    max_asymmetry = float(np.max(np.abs(sub - sub.T)))
    if max_asymmetry > 1e-3:
        print(
            f"Warning: matrix is not symmetric (maximum |R-R.T| = {max_asymmetry:.3g}); "
            "using (R + R.T)/2 for the decomposition.",
            file=sys.stderr,
        )
    sub = (sub + sub.T) / 2.0

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(sub)
    except np.linalg.LinAlgError as exc:
        print(f"Error: eigendecomposition failed: {exc}", file=sys.stderr)
        return 2

    # eigh returns eigenvalues in ascending order; reorder descending to define PC1, PC2, ...
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if args.pc > len(eigenvalues):
        print(
            f"Error: requested PC{args.pc}, but only {len(eigenvalues)} components are available.",
            file=sys.stderr,
        )
        return 2

    selected = eigenvectors[:, args.pc - 1].copy()
    if not args.keep_sign:
        # The sign of any eigenvector is arbitrary; choose a reproducible display convention.
        pivot = int(np.argmax(np.abs(selected)))
        if selected[pivot] < 0:
            selected *= -1.0

    vector_full = np.full(n_original, np.nan, dtype=float)
    vector_full[valid] = selected

    try:
        np.savetxt(output_path, vector_full, fmt="%.10g")
    except OSError as exc:
        print(f"Error writing output '{output_path}': {exc}", file=sys.stderr)
        return 2

    # Eigenvalue proportions summarize variance captured for a standard PCA based
    # on this correlation matrix. Tiny negative values may arise from numerical error.
    total = float(np.sum(eigenvalues))
    if total > 0:
        proportions = eigenvalues / total
    else:
        proportions = np.full_like(eigenvalues, np.nan)

    if args.eigenvalues:
        try:
            with open(args.eigenvalues, "w", encoding="utf-8") as handle:
                handle.write("component\teigenvalue\tvariance_fraction\tcumulative_fraction\n")
                cumulative = 0.0
                for i, (val, frac) in enumerate(zip(eigenvalues, proportions), start=1):
                    if np.isfinite(frac):
                        cumulative += float(frac)
                    frac_text = f"{frac:.10g}" if np.isfinite(frac) else "nan"
                    handle.write(f"PC{i}\t{val:.10g}\t{frac_text}\t{cumulative:.10g}\n")
        except OSError as exc:
            print(f"Error writing eigenvalue table '{args.eigenvalues}': {exc}", file=sys.stderr)
            return 2

    variance_fraction = proportions[args.pc - 1]
    print(f"Input matrix: {matrix_path}")
    print(f"Original matrix dimensions: {n_original} x {n_original}")
    print(f"Bins included in decomposition: {n_valid}")
    print(f"Selected component: PC{args.pc}")
    print(f"Eigenvalue: {eigenvalues[args.pc - 1]:.8g}")
    if np.isfinite(variance_fraction):
        print(f"Fraction of total eigenvalue sum: {variance_fraction:.4%}")
    print(f"Vector saved to: {output_path}")
    if args.eigenvalues:
        print(f"Eigenvalue table saved to: {args.eigenvalues}")
    print("Bins excluded from the decomposition remain as NaN in the output.")
    if not args.keep_sign:
        print("Sign convention: largest absolute loading is positive; PC signs are arbitrary.")
    else:
        print("The eigensolver's sign was retained; PC signs are arbitrary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
