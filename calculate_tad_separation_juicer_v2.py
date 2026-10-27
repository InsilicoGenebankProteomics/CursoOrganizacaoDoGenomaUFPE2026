#!/usr/bin/env python3
"""Calculate a Juicer-based, HiCExplorer-style TAD-separation score.

This script does NOT require HiCExplorer. It uses Java + Juicer Tools to dump
KR-balanced (or another requested normalization) contacts from one chromosome
of a .hic file, then calculates distance-specific z-scores and the mean z-score
of contacts crossing each bin boundary for multiple window depths.

Important: this follows the published/core definition of the TAD-separation
score, but it is an independent implementation, not a bit-for-bit reproduction
of HiCExplorer. Masking and matrix-correction details can differ.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Este script precisa do NumPy. Instale apenas essa dependência com: "
        "python3 -m pip install numpy"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Calcula um TAD-separation score estilo HiCExplorer diretamente de "
            "um .hic usando Juicer Tools + NumPy, sem instalar HiCExplorer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("hic", help="Arquivo .hic de entrada")
    p.add_argument("--juicer-jar", required=True, help="Caminho para juicer_tools.jar")
    p.add_argument("--chromosome", required=True, help="ID exato do cromossomo no .hic (ex.: chr1 ou Chr1)")
    p.add_argument("--resolution", type=int, required=True, help="Resolução em bp, que deve existir no .hic (ex.: 10000)")
    p.add_argument("--normalization", choices=["KR", "VC", "VC_SQRT", "NONE"], default="KR",
                   help="Normalização usada ao exportar os contatos com Juicer Tools")
    p.add_argument("--chrom-length", type=int, default=None,
                   help="Comprimento exato do cromossomo em bp; recomendado para manter também bins finais sem contatos")
    p.add_argument("--chrom-sizes", default=None,
                   help="Arquivo de tamanhos cromossômicos (duas colunas: ID e comprimento em bp); alternativa a --chrom-length")
    p.add_argument("--min-depth", type=int, default=None,
                   help="Janela mínima em bp para cada lado da fronteira; por padrão segue proporções usuais do HiCExplorer")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Janela máxima em bp para cada lado da fronteira")
    p.add_argument("--step", type=int, default=None,
                   help="Passo inicial em bp entre escalas (o incremento cresce como step * x**1.5)")
    p.add_argument("--out-prefix", default="tad_separation",
                   help="Prefixo dos arquivos de saída")
    p.add_argument("--java", default="java", help="Executável Java")
    p.add_argument("--heap", default="8g", help="Memória Java para o dump, por exemplo 4g ou 8g")
    p.add_argument("--keep-dump", action="store_true", help="Preserva o arquivo de contatos exportado pelo Juicer Tools")
    p.add_argument("--no-plot", action="store_true", help="Não gera os gráficos PNG e PDF")
    p.add_argument("--plot-dpi", type=int, default=200, help="Resolução do gráfico PNG")
    return p.parse_args()


def read_chrom_length(chrom: str, chrom_length: Optional[int], chrom_sizes: Optional[str]) -> Optional[int]:
    if chrom_length is not None:
        if chrom_length <= 0:
            raise ValueError("--chrom-length precisa ser um inteiro positivo")
        if chrom_sizes:
            raise ValueError("Use --chrom-length ou --chrom-sizes, não ambos")
        return chrom_length
    if chrom_sizes:
        with open(chrom_sizes, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 2 and fields[0] == chrom:
                    length = int(fields[1])
                    if length <= 0:
                        raise ValueError(f"Comprimento inválido para {chrom}: {length}")
                    return length
        raise ValueError(f"O cromossomo {chrom!r} não foi encontrado em {chrom_sizes}")
    return None


def iter_dump_rows(path: str):
    """Read Juicer's sparse dump (bin_start_1, bin_start_2, value)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            fields = s.split()
            if len(fields) < 3:
                continue
            try:
                pos1 = int(float(fields[0]))
                pos2 = int(float(fields[1]))
                value = float(fields[2])
            except ValueError:
                # Skip a possible header/log line.
                continue
            if not math.isfinite(value):
                continue
            if pos1 < 0 or pos2 < 0:
                continue
            yield pos1, pos2, value


def default_depths(resolution: int) -> Tuple[int, int, int]:
    # Matches the general default scale factors documented by HiCExplorer.
    if resolution < 1000:
        return resolution * 30, resolution * 60, resolution * 4
    if resolution < 20000:
        return resolution * 10, resolution * 40, resolution * 2
    return resolution * 5, resolution * 10, resolution * 2


def make_depths(min_depth: int, max_depth: int, step: int) -> List[int]:
    """Use the same progression described by hicFindTADs: min + step*x^1.5."""
    depths: List[int] = []
    x = 0
    while True:
        d = min_depth + int(step * (x ** 1.5))
        if d > max_depth:
            break
        if not depths or d > depths[-1]:
            depths.append(d)
        x += 1
    if not depths:
        depths = [min_depth]
    return depths


def calculate_scores(
    diagonals: Dict[int, np.ndarray], n_bins: int, resolution: int, depths_bp: List[int],
    max_z_distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (score_by_window, mean_score), indexed by boundary/bin.

    For each genomic distance d, z-score values on that diagonal. For a cut at
    bin k and a window depth w, average the z-scores of every matrix entry in
    rows [k-w, k) and columns [k, k+w), clipping to chromosome bounds.
    """
    z_diagonals: Dict[int, np.ndarray] = {}
    for d in range(1, min(max_z_distance, n_bins - 1) + 1):
        values = diagonals.get(d)
        if values is None:
            values = np.zeros(n_bins - d, dtype=np.float64)
        else:
            values = values.astype(np.float64, copy=False)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if not math.isfinite(std) or std == 0:
            z = np.zeros_like(values)
        else:
            z = (values - mean) / std
        # Prefix sum supports fast averages of parts of the diagonal.
        z_diagonals[d] = np.concatenate(([0.0], np.cumsum(z, dtype=np.float64)))

    all_scores = np.full((n_bins, len(depths_bp)), np.nan, dtype=np.float64)
    for col, depth_bp in enumerate(depths_bp):
        w = max(1, int(depth_bp // resolution))
        for k in range(n_bins):
            left_start = max(0, k - w)
            right_end = min(n_bins, k + w)
            if left_start >= k or right_end <= k:
                continue
            total = 0.0
            count = 0
            dmax = min(right_end - left_start - 1, max_z_distance, n_bins - 1)
            for d in range(1, dmax + 1):
                # i is in [left_start, k-1], and i+d is in [k, right_end-1].
                lo = max(left_start, k - d, 0)
                hi = min(k - 1, right_end - d - 1, n_bins - d - 1)
                if lo > hi:
                    continue
                pref = z_diagonals.get(d)
                if pref is None:
                    continue
                total += float(pref[hi + 1] - pref[lo])
                count += hi - lo + 1
            if count:
                all_scores[k, col] = total / count

    valid_counts = np.sum(np.isfinite(all_scores), axis=1)
    score_sums = np.nansum(all_scores, axis=1)
    mean_scores = np.full(n_bins, np.nan, dtype=np.float64)
    valid_rows = valid_counts > 0
    mean_scores[valid_rows] = score_sums[valid_rows] / valid_counts[valid_rows]
    return all_scores, mean_scores


def plot_scores(
    chromosome: str,
    resolution: int,
    chrom_length: int,
    mean_scores: np.ndarray,
    out_prefix: str,
    dpi: int = 200,
) -> Tuple[str, str]:
    """Save a black-bar plot of mean TAD-separation score across the chromosome."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Para gerar o gráfico, instale Matplotlib com: python3 -m pip install matplotlib. "
            "Você também pode executar novamente com --no-plot para salvar apenas os dados."
        ) from exc

    valid = np.isfinite(mean_scores)
    if not np.any(valid):
        raise RuntimeError("Não há scores finitos para plotar.")

    centers_mb = (np.arange(len(mean_scores))[valid] + 0.5) * resolution / 1_000_000.0
    widths_mb = resolution / 1_000_000.0
    values = mean_scores[valid]
    prefix = Path(out_prefix)
    png_path = str(prefix) + "_tad_separation.png"
    pdf_path = str(prefix) + "_tad_separation.pdf"

    # Size scales gently with chromosome length, with sensible upper bounds.
    width_inches = min(18.0, max(10.0, chrom_length / 1_000_000.0 * 0.22))
    fig, ax = plt.subplots(figsize=(width_inches, 5.2), constrained_layout=True)
    ax.bar(centers_mb, values, width=widths_mb, color="black", edgecolor="black", linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Posição ao longo do cromossomo (Mb)")
    ax.set_ylabel("TAD-separation score (média entre escalas)")
    ax.set_title(f"TAD-separation score — {chromosome}")
    ax.set_xlim(0, chrom_length / 1_000_000.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    hic_path = os.path.abspath(args.hic)
    jar_path = os.path.abspath(args.juicer_jar)
    if not os.path.isfile(hic_path):
        sys.exit(f"ERRO: arquivo .hic não encontrado: {hic_path}")
    if not os.path.isfile(jar_path):
        sys.exit(f"ERRO: Juicer Tools JAR não encontrado: {jar_path}")
    if args.resolution <= 0:
        sys.exit("ERRO: --resolution precisa ser positivo")
    if shutil.which(args.java) is None:
        sys.exit(f"ERRO: não encontrei o executável Java: {args.java}")

    try:
        chrom_length = read_chrom_length(args.chromosome, args.chrom_length, args.chrom_sizes)
    except Exception as exc:
        sys.exit(f"ERRO: {exc}")

    dmin, dmax, dstep = default_depths(args.resolution)
    min_depth = args.min_depth if args.min_depth is not None else dmin
    max_depth = args.max_depth if args.max_depth is not None else dmax
    step = args.step if args.step is not None else dstep
    if min_depth < 3 * args.resolution:
        sys.exit(f"ERRO: --min-depth deve ser pelo menos 3 × resolução ({3 * args.resolution} bp)")
    if max_depth < 5 * args.resolution:
        sys.exit(f"ERRO: --max-depth deve ser pelo menos 5 × resolução ({5 * args.resolution} bp)")
    if max_depth <= min_depth:
        sys.exit("ERRO: --max-depth deve ser maior que --min-depth")
    if step < args.resolution:
        sys.exit("ERRO: --step deve ser >= resolução")

    depths = make_depths(min_depth, max_depth, step)
    max_z_distance = max(1, int(math.ceil((2.5 * max_depth) / args.resolution)))
    print(f"Cromossomo: {args.chromosome}; resolução: {args.resolution:,} bp")
    print(f"Normalização Juicer: {args.normalization}; janelas (bp): {', '.join(map(str, depths))}")
    print("Exportando contatos com Juicer Tools. Isso pode demorar para matrizes grandes...", flush=True)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="tadsep_juicer_")
    temp_dir = temp_dir_obj.name
    dump_path = os.path.join(temp_dir, "observed_sparse.txt")
    cmd = [
        args.java, f"-Xmx{args.heap}", "-jar", jar_path, "dump", "observed",
        args.normalization, hic_path, args.chromosome, args.chromosome,
        "BP", str(args.resolution), dump_path,
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        temp_dir_obj.cleanup()
        sys.exit(f"ERRO ao executar Juicer Tools: {exc}")
    if proc.returncode != 0 or not os.path.isfile(dump_path):
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        temp_dir_obj.cleanup()
        sys.exit(
            "ERRO: Juicer Tools não conseguiu exportar a matriz. Confira se o nome do cromossomo "
            "e a resolução existem no .hic e se a normalização selecionada está disponível."
        )
    if os.path.getsize(dump_path) == 0:
        print(proc.stderr, file=sys.stderr)
        temp_dir_obj.cleanup()
        sys.exit("ERRO: o dump ficou vazio. A resolução, cromossomo ou normalização pode não estar disponível.")

    max_seen_bin = -1
    row_count = 0
    for pos1, pos2, value in iter_dump_rows(dump_path):
        max_seen_bin = max(max_seen_bin, pos1 // args.resolution, pos2 // args.resolution)
        row_count += 1
    if row_count == 0:
        temp_dir_obj.cleanup()
        sys.exit("ERRO: não encontrei linhas numéricas no dump do Juicer Tools.")
    if chrom_length is None:
        # Approximation only: a final bin without any reported contact may be omitted.
        chrom_length = (max_seen_bin + 1) * args.resolution
        print(
            "AVISO: --chrom-length/--chrom-sizes não foi informado. O comprimento foi estimado "
            "pelo último bin com contatos; bins finais sem contatos podem ficar de fora.",
            file=sys.stderr,
        )
    n_bins = int(math.ceil(chrom_length / args.resolution))
    if max_seen_bin >= n_bins:
        n_bins = max_seen_bin + 1
        chrom_length = n_bins * args.resolution
        print("AVISO: o dump contém coordenadas além do comprimento informado; ajustei o número de bins.", file=sys.stderr)

    max_z_distance = min(max_z_distance, n_bins - 1)
    # Store only diagonals needed for the score; memory is O(number of bins × window width),
    # not O(number of bins squared).
    diagonals: Dict[int, np.ndarray] = {
        d: np.zeros(n_bins - d, dtype=np.float64)
        for d in range(1, max_z_distance + 1)
    }
    for pos1, pos2, value in iter_dump_rows(dump_path):
        b1 = pos1 // args.resolution
        b2 = pos2 // args.resolution
        if b1 >= n_bins or b2 >= n_bins or b1 == b2:
            continue
        i, j = (b1, b2) if b1 < b2 else (b2, b1)
        d = j - i
        if d <= max_z_distance:
            # Juicer's dump is sparse. Unreported contacts are treated as zero.
            diagonals[d][i] = value

    all_scores, mean_scores = calculate_scores(
        diagonals, n_bins, args.resolution, depths, max_z_distance
    )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = str(prefix) + "_tad_separation.tsv"
    bedgraph_path = str(prefix) + "_score.bedgraph"
    with open(tsv_path, "w", encoding="utf-8") as out:
        columns = [f"score_{d}bp" for d in depths]
        out.write("chrom\tstart\tend\t" + "\t".join(columns) + "\tmean_score\n")
        for i in range(n_bins):
            start = i * args.resolution
            end = min((i + 1) * args.resolution, chrom_length)
            values = ["NA" if not np.isfinite(v) else f"{v:.8g}" for v in all_scores[i]]
            mean_value = "NA" if not np.isfinite(mean_scores[i]) else f"{mean_scores[i]:.8g}"
            out.write(f"{args.chromosome}\t{start}\t{end}\t" + "\t".join(values) + f"\t{mean_value}\n")
    with open(bedgraph_path, "w", encoding="utf-8") as out:
        for i, score in enumerate(mean_scores):
            if not np.isfinite(score):
                continue
            start = i * args.resolution
            end = min((i + 1) * args.resolution, chrom_length)
            out.write(f"{args.chromosome}\t{start}\t{end}\t{score:.8g}\n")

    if not args.no_plot:
        try:
            png_path, pdf_path = plot_scores(
                args.chromosome, args.resolution, chrom_length, mean_scores,
                str(prefix), dpi=args.plot_dpi,
            )
            print(f"Gráfico de barras PNG: {png_path}")
            print(f"Gráfico de barras PDF: {pdf_path}")
        except Exception as exc:
            temp_dir_obj.cleanup()
            sys.exit(f"ERRO ao gerar o gráfico: {exc}")

    if args.keep_dump:
        dest = str(prefix) + "_juicer_dump.txt"
        shutil.copy2(dump_path, dest)
        print(f"Dump preservado: {dest}")

    temp_dir_obj.cleanup()
    print("Concluído.")
    print(f"Score por escala + média: {tsv_path}")
    print(f"Track bedGraph (média das escalas): {bedgraph_path}")
    print("Interpretação: mínimos locais negativos podem indicar fronteiras; o score não chama TADs por si só.")
    print("Nota metodológica: este é um cálculo independente estilo HiCExplorer, não uma reprodução bit a bit do HiCExplorer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
