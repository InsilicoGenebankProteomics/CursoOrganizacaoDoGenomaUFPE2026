#!/usr/bin/env python3
"""Plot Hi-C distance-decay curves from Juicer Tools dump output.

-av   : average raw Juicer expected-value vectors
-avn  : normalize each expected vector to sum to 1, then average
-avp  : sum observed contacts by separation / possible bin pairs, then average
-avpn : calculate P(s), normalize each curve to sum to 1, then average

Optional --plot-random-coil overlays a theoretical s^(-3/2) reference slope
on the same log-log axes, scaled to the first plotted curve at a reference
separation. For -avp/-avpn, a chromosome length is still required to calculate
the number of possible bin pairs; supply --chrom-sizes or --chrom-length.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_expected(filename, bin_size):
    try:
        vals = np.loadtxt(filename, ndmin=1).astype(float).reshape(-1)
    except ValueError as exc:
        raise ValueError(
            f"{filename}: expected a numeric, one-column expected-values file. "
            "Check that the Juicer dump succeeded (the file must not begin with 'Usage:' or an error)."
        ) from exc
    dist = np.arange(len(vals), dtype=float) * bin_size
    vals[~np.isfinite(vals) | (vals <= 0)] = np.nan
    keep = dist > 0
    return dist[keep], vals[keep]


def read_sizes(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            fields = s.split()
            if len(fields) < 2:
                continue
            try:
                out[fields[0]] = int(fields[1])
            except ValueError:
                continue  # tolerate an optional header
    if not out:
        raise ValueError(f"Could not read chromosome sizes from {path}")
    return out


def npossible_curve(chrom_length, bin_size):
    """Return distance (bp), possible-pair count, and number of bins.

    At separation k bins, a chromosome with N bins has N-k possible pairs.
    """
    n_bins = int(np.ceil(chrom_length / bin_size))
    if n_bins < 2:
        raise ValueError(
            f"Chromosome length ({chrom_length}) is too short for bin size {bin_size}."
        )
    separation = np.arange(1, n_bins)
    distance = separation.astype(float) * bin_size
    possible_pairs = (n_bins - separation).astype(float)
    return distance, possible_pairs, n_bins


def chromosome_length(filename, args, sizes_cache):
    if args.chrom_length is not None:
        return args.chrom_length
    if not args.chrom_sizes:
        raise ValueError(
            f"{filename}: -avp/-avpn require chromosome length. Supply --chrom-sizes FILE "
            "or --chrom-length BP for files from a single chromosome."
        )
    if args.chrom_sizes not in sizes_cache:
        sizes_cache[args.chrom_sizes] = read_sizes(args.chrom_sizes)
    sizes = sizes_cache[args.chrom_sizes]
    if args.chromosome:
        if args.chromosome not in sizes:
            raise ValueError(f"Chromosome {args.chromosome!r} not found in {args.chrom_sizes}")
        return sizes[args.chromosome]
    base = os.path.basename(filename)
    hits = [chrom for chrom in sizes if chrom in base]
    if hits:
        # Prefer the longest match, e.g. chr10 rather than chr1.
        return sizes[max(hits, key=len)]
    if len(sizes) == 1:
        return next(iter(sizes.values()))
    raise ValueError(
        f"Cannot infer chromosome from filename {base!r}. Include its chromosome ID in the filename "
        "or pass --chromosome CHR."
    )


def read_observed_ps(filename, bin_size, chrom_length):
    """Read Juicer's 3 columns: bin1_start_bp, bin2_start_bp, observed value."""
    try:
        data = np.loadtxt(filename, ndmin=2).astype(float)
    except ValueError as exc:
        raise ValueError(
            f"{filename}: expected numeric three-column observed data. Check for Juicer dump errors/usage text."
        ) from exc
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{filename}: expected 3 columns (bin1_start, bin2_start, observed_value).")

    p1, p2, contacts = data[:, 0], data[:, 1], data[:, 2]
    good = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(contacts) & (contacts >= 0)
    p1, p2, contacts = p1[good], p2[good], contacts[good]

    # Juicer reports genomic coordinates in bp. Convert coordinate separation
    # to a bin offset; do not multiply the bp coordinates by bin_size.
    sep_bp = np.abs(p1 - p2)
    sep_float = sep_bp / bin_size
    sep = np.rint(sep_float).astype(int)
    if len(sep_float) and np.any(np.abs(sep_float - sep) > 1e-6):
        raise ValueError(
            f"{filename}: coordinates are not multiples of --bin-size {bin_size}; "
            "check that the dump used BP at this resolution."
        )

    distances, possible_pairs, n_bins = npossible_curve(chrom_length, bin_size)
    if len(sep) and np.max(sep) >= n_bins:
        raise ValueError(
            f"{filename}: a contact separation exceeds the supplied chromosome length. "
            "Check --chrom-sizes/--chromosome/--chrom-length."
        )

    # Exclude the diagonal (s=0); absent sparse-matrix cells contribute zero.
    off_diag = sep > 0
    sums = np.bincount(sep[off_diag], weights=contacts[off_diag], minlength=n_bins)[:n_bins]
    ps = sums[1:] / possible_pairs
    return distances, ps


def normalize(values):
    values = np.asarray(values, dtype=float)
    total = np.nansum(values[np.isfinite(values) & (values > 0)])
    if total <= 0:
        raise ValueError("Cannot normalize a curve with no positive finite values.")
    return values / total


def combine(curves):
    # Align by genomic distance, not by array position. This matters when
    # curves differ in chromosome length or have different valid ranges.
    x = np.unique(np.concatenate([c[0] for c in curves]))
    mat = np.full((len(curves), len(x)), np.nan)
    for i, (dist, vals) in enumerate(curves):
        idx = np.searchsorted(x, dist)
        mat[i, idx] = vals
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(mat, axis=0)
    sd = np.zeros_like(mean)
    for j in range(mat.shape[1]):
        v = mat[:, j]
        v = v[np.isfinite(v)]
        if len(v) > 1:
            sd[j] = np.std(v, ddof=1)
    return x, mean, sd


def add_group(groups, files, kind, do_norm, args, size_cache):
    curves = []
    for f in files:
        if kind == 'expected':
            dist, vals = read_expected(f, args.bin_size)
        else:
            length = chromosome_length(f, args, size_cache)
            dist, vals = read_observed_ps(f, args.bin_size, length)
        if do_norm:
            vals = normalize(vals)
        curves.append((dist, vals))
    x, mean, sd = combine(curves)
    groups.append(dict(x=x, mean=mean, sd=sd, files=files, kind=kind, normalized=do_norm))


def add_random_coil_reference(ax, groups, reference_distance=None):
    """Overlay a scaled s^(-3/2) random-coil reference on the primary axes.

    The prefactor of the theoretical power law is not specified, so the curve
    is anchored to the first plotted curve at a reference genomic separation.
    If no reference distance is supplied, the geometric mean of the first
    curve's valid distance range is used.
    """
    first = groups[0]
    x = np.asarray(first['x'], dtype=float)
    y = np.asarray(first['mean'], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_valid, y_valid = x[valid], y[valid]
    if len(x_valid) < 2:
        raise ValueError(
            "Cannot plot the random-coil reference: the first curve needs at least "
            "two positive finite data points."
        )
    order = np.argsort(x_valid)
    x_valid, y_valid = x_valid[order], y_valid[order]

    if reference_distance is None:
        reference_distance = float(np.sqrt(x_valid[0] * x_valid[-1]))
    if reference_distance <= 0:
        raise ValueError('--random-coil-reference-distance must be positive.')
    if reference_distance < x_valid[0] or reference_distance > x_valid[-1]:
        raise ValueError(
            f'Reference distance {reference_distance:g} bp is outside the first curve range '
            f'({x_valid[0]:g} to {x_valid[-1]:g} bp). Choose a distance within this range.'
        )

    # Interpolate in log-log space so the anchor lies on the plotted curve.
    y_reference = float(np.exp(np.interp(
        np.log(reference_distance), np.log(x_valid), np.log(y_valid)
    )))
    x_reference_curve = np.geomspace(x_valid[0], x_valid[-1], 300)
    y_reference_curve = y_reference * (x_reference_curve / reference_distance) ** (-1.5)
    ax.plot(
        x_reference_curve,
        y_reference_curve,
        linestyle='--',
        linewidth=1.8,
        color='black',
        alpha=0.8,
        label=f'Random-coil reference $s^{{-3/2}}$ (anchored at {reference_distance:g} bp)',
    )


def main():
    p = argparse.ArgumentParser(description='Plot Hi-C distance-dependent decay from Juicer Tools output.')
    p.add_argument('-av', nargs='+', action='append', default=[], metavar='FILE', help='Average expected-value files; repeat -av for separate curves.')
    p.add_argument('-avn', nargs='+', action='append', default=[], metavar='FILE', help='Normalize each expected curve, then average.')
    p.add_argument('-avp', nargs='+', action='append', default=[], metavar='FILE', help='Calculate P(s) from observed files, then average.')
    p.add_argument('-avpn', nargs='+', action='append', default=[], metavar='FILE', help='Calculate P(s), normalize each curve, then average.')
    p.add_argument('--bin-size', type=int, default=100000, help='Juicer BP bin size (default: 100000).')
    p.add_argument('--chrom-sizes', help='Two-column chromosome-size file; chromosome ID is inferred from each observed filename.')
    p.add_argument('--chromosome', help='Explicit chromosome ID for observed files (used with --chrom-sizes).')
    p.add_argument('--chrom-length', type=int, help='Chromosome length in bp; alternative to --chrom-sizes for one chromosome.')
    p.add_argument(
        '--plot-random-coil', '--plot-npossible', dest='plot_random_coil', action='store_true',
        help='Overlay a scaled random-coil reference slope s^(-3/2) on the main axes. '
             '--plot-npossible is retained as a deprecated alias.'
    )
    p.add_argument(
        '--random-coil-reference-distance', type=float, default=None,
        help='Genomic separation in bp where the s^(-3/2) reference is anchored to the first plotted curve; '
             'default: geometric mean of the first curve valid distance range.'
    )
    p.add_argument('--output', default='hic_decay.pdf', help='Output PDF filename.')
    p.add_argument('--png', default='hic_decay.png', help='Output PNG filename.')
    p.add_argument('--no-show', action='store_true', help='Save files without opening a plot window.')
    a = p.parse_args()
    if a.bin_size <= 0:
        p.error('--bin-size must be positive')
    if a.chrom_length is not None and a.chrom_sizes:
        p.error('Use either --chrom-length or --chrom-sizes, not both.')
    if a.chrom_length is not None and a.chrom_length <= 0:
        p.error('--chrom-length must be positive')
    if a.random_coil_reference_distance is not None and a.random_coil_reference_distance <= 0:
        p.error('--random-coil-reference-distance must be positive.')

    groups, cache = [], {}
    try:
        for g in a.av: add_group(groups, g, 'expected', False, a, cache)
        for g in a.avn: add_group(groups, g, 'expected', True, a, cache)
        for g in a.avp: add_group(groups, g, 'probability', False, a, cache)
        for g in a.avpn: add_group(groups, g, 'probability', True, a, cache)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if not groups:
        p.error('Choose at least one option: -av, -avn, -avp, or -avpn.')

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for g in groups:
        files = g['files']
        label = os.path.basename(files[0]) if len(files) == 1 else f"Average ({len(files)})"
        label += ' expected' if g['kind'] == 'expected' else ' P(s)'
        if g['normalized']: label += ' normalized'
        x, m, sd = g['x'], g['mean'], g['sd']
        y = np.where(np.isfinite(m) & (m > 0), m, np.nan)
        ax.plot(x, y, linewidth=2, label=label)
        if len(files) > 1:
            upper = m + sd
            lower = np.maximum(m - sd, np.finfo(float).tiny)
            ok = np.isfinite(m) & (m > 0) & np.isfinite(sd) & (upper > 0)
            ax.fill_between(x, np.where(ok, lower, np.nan), np.where(ok, upper, np.nan), alpha=0.20)

    if a.plot_random_coil:
        try:
            add_random_coil_reference(ax, groups, a.random_coil_reference_distance)
        except ValueError as exc:
            p.error(str(exc))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Genomic distance (bp)')
    ax.set_ylabel('Contact frequency (expected signal or contacts per possible pair)')
    ax.set_title('Hi-C distance-dependent decay')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.output, bbox_inches='tight')
    fig.savefig(a.png, dpi=300, bbox_inches='tight')
    if a.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
