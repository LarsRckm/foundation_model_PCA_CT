"""
Cumulative-variance violin plot of DINOv2 PCA scores (interactive).

Layout (inspired by a split-violin "comparison of methods" plot):
  x-axis : PC1, PC2, ... up to the PC where cumulative variance first exceeds
           --var_threshold (default 99%).
  y-axis : cumulative explained variance (%).
  Each PC is drawn as a SPLIT VIOLIN (left = real, right = synthetic) of that
  PC's scores, vertically centered at the PC's cumulative-variance value, so the
  violins climb along the cumulative-variance curve.

Interactivity:
  Click on a PC's violin -> that PC opens in a separate window showing its full
  score distribution (real vs. synthetic, with explained/cumulative variance).
  (Needs an interactive Matplotlib backend, e.g. TkAgg/QtAgg. The static main
  figure is always saved as PNG regardless.)

Model loading is identical to cluster_ct_scans.py / pca_analysis_visualization.py
(facebook/dinov2-small, patch size 14). The PCA basis is fit on the REAL patch
tokens only; the synthetic tokens are projected onto the real axes (scalar product
after subtracting the real mean), like PCA_comparison. Real is centred at the origin,
and the cumulative-variance curve is the explained variance of the real data.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

# Orange / purple split to mirror the reference figure.
COLOR_REAL = "#d95f0e"
COLOR_SYNTH = "#5e3c99"


PATCH_SIZE = 14


def _ensure_dir(path):
    """Create the parent directory of ``path`` if needed (results folders)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def processed_image_to_pil(pixel_values, processor):
    """Undo the processor normalization so the displayed image matches the patch grid."""
    pv = pixel_values[0]  # (3, H, W)
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    img = (pv * std + mean).clamp(0, 1)
    arr = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def extract(model, processor, image):
    """Return (patch_embeddings [N, D], grid_h, grid_w, processed_pil_image)."""
    inputs = processor(images=image, return_tensors="pt")
    h, w = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
    grid_h, grid_w = h // PATCH_SIZE, w // PATCH_SIZE
    with torch.no_grad():
        outputs = model(**inputs)
    # Drop the CLS token (index 0); keep the patch tokens.
    patches = outputs.last_hidden_state[0, 1:, :].numpy()
    proc_img = processed_image_to_pil(inputs["pixel_values"], processor)
    return patches, grid_h, grid_w, proc_img


def _density(data, grid):
    """KDE on a grid, with a flat fallback for (near-)constant data."""
    if data.size > 2 and np.std(data) > 1e-9:
        return gaussian_kde(data)(grid)
    d = np.zeros_like(grid)
    d[len(d) // 2] = 1.0
    return d


def _overlay_pc_heatmap(ax, proc_img, score_map, vmax, title):
    """Variant B: grayscale image + semi-transparent diverging heatmap of the
    per-patch PC scores. Red/blue = the two poles of the component; the score
    sign is arbitrary, so both ends matter."""
    gray = np.asarray(proc_img.convert("L"))
    h, w = gray.shape
    extent = [-0.5, w - 0.5, h - 0.5, -0.5]
    ax.imshow(gray, cmap="gray", aspect="auto", extent=extent)
    im = ax.imshow(score_map, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   alpha=0.6, interpolation="nearest", aspect="auto", extent=extent)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def draw_split_violin(ax, x_center, y_center, real_s, synth_s,
                      half_height, half_width,
                      color_real=COLOR_REAL, color_synth=COLOR_SYNTH):
    """
    Draw a split violin centered at (x_center, y_center).
    The score axis is vertical (mapped into +-half_height around y_center);
    the density is horizontal (real on the left, synthetic on the right).
    Returns nothing; band geometry for click-hit-testing is handled by caller.
    """
    all_s = np.concatenate([real_s, synth_s])
    lo, hi = np.percentile(all_s, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    grid = np.linspace(lo, hi, 120)

    dens_r = _density(real_s, grid)
    dens_s = _density(synth_s, grid)
    maxd = max(dens_r.max(), dens_s.max(), 1e-12)
    wr = dens_r / maxd * half_width
    ws = dens_s / maxd * half_width

    vcenter = 0.5 * (lo + hi)
    halfrange = 0.5 * (hi - lo)
    y = y_center + half_height * (grid - vcenter) / halfrange

    ax.fill_betweenx(y, x_center - wr, x_center, color=color_real,
                     alpha=0.85, lw=0.4, edgecolor="white")
    ax.fill_betweenx(y, x_center, x_center + ws, color=color_synth,
                     alpha=0.85, lw=0.4, edgecolor="white")


def make_detail_window(pc_idx, scores, real_mask, synth_mask, evr, cum,
                       grids, proc_imgs, n_real):
    """Separate window for one PC: score distribution (left) plus the PC's
    patch-score heatmap over both images (right, variant B)."""
    real_s = scores[real_mask, pc_idx]
    synth_s = scores[synth_mask, pc_idx]
    lo, hi = np.percentile(scores[:, pc_idx], [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)

    (gh_r, gw_r), (gh_s, gw_s) = grids
    real_map = scores[:n_real, pc_idx].reshape(gh_r, gw_r)
    synth_map = scores[n_real:, pc_idx].reshape(gh_s, gw_s)
    vmax = max(np.abs(real_map).max(), np.abs(synth_map).max(), 1e-9)

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"PC{pc_idx + 1} - distribution & heatmap")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])

    ax_kde = fig.add_subplot(gs[:, 0])
    for data, color, name in [(real_s, COLOR_REAL, "Real"),
                              (synth_s, COLOR_SYNTH, "Synthetic")]:
        dens = _density(data, grid)
        ax_kde.plot(grid, dens, color=color, lw=1.8, label=name)
        ax_kde.fill_between(grid, dens, color=color, alpha=0.25)
        ax_kde.axvline(np.mean(data), color=color, ls="--", lw=1)
    ax_kde.set_title(f"PC{pc_idx + 1}   |   explained variance {evr[pc_idx]*100:.2f}%"
                     f"   |   cumulative {cum[pc_idx]*100:.2f}%")
    ax_kde.set_xlabel(f"PC{pc_idx + 1} score")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()

    ax_r = fig.add_subplot(gs[0, 1])
    im = _overlay_pc_heatmap(ax_r, proc_imgs[0], real_map, vmax, "Real")
    ax_s = fig.add_subplot(gs[1, 1])
    _overlay_pc_heatmap(ax_s, proc_imgs[1], synth_map, vmax, "Synthetic")
    fig.colorbar(im, ax=[ax_r, ax_s], fraction=0.05, pad=0.02,
                 label=f"PC{pc_idx + 1} score (red/blue = the two poles)")

    # Show non-blocking so the main window's event loop keeps running.
    fig.show()
    fig.canvas.draw_idle()


def save_pc_heatmaps(pcs, scores, n_real, grids, proc_imgs, evr, path):
    """Variant B as a static figure: rows = PCs, columns = real / synthetic."""
    (gh_r, gw_r), (gh_s, gw_s) = grids
    n = len(pcs)
    fig = plt.figure(figsize=(7, 3.2 * n), constrained_layout=True)
    gs = fig.add_gridspec(n, 2)
    for row, pc in enumerate(pcs):
        real_map = scores[:n_real, pc].reshape(gh_r, gw_r)
        synth_map = scores[n_real:, pc].reshape(gh_s, gw_s)
        vmax = max(np.abs(real_map).max(), np.abs(synth_map).max(), 1e-9)
        ax_r = fig.add_subplot(gs[row, 0])
        im = _overlay_pc_heatmap(ax_r, proc_imgs[0], real_map, vmax,
                                 f"PC{pc + 1} ({evr[pc]*100:.1f}%) - Real")
        ax_s = fig.add_subplot(gs[row, 1])
        _overlay_pc_heatmap(ax_s, proc_imgs[1], synth_map, vmax, "Synthetic")
        fig.colorbar(im, ax=[ax_r, ax_s], fraction=0.05, pad=0.02)
    fig.suptitle("Per-PC patch-score heatmaps (variant B)")
    _ensure_dir(path)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved per-PC heatmaps to {path}")


def parse_pc_list(spec):
    """Parse '1-6' or '1,3,5' (1-based) into a sorted list of 0-based indices."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i - 1 for i in out if i >= 1)


def make_overall_detail_window(real_s, synth_s, k, total_var):
    """Separate window for the pooled (all-PC, standardized) score distribution."""
    lo, hi = np.percentile(np.concatenate([real_s, synth_s]), [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.canvas.manager.set_window_title("Overall PC distribution")

    for data, color, name in [(real_s, COLOR_REAL, "Real"),
                              (synth_s, COLOR_SYNTH, "Synthetic")]:
        dens = _density(data, grid)
        ax.plot(grid, dens, color=color, lw=1.8, label=name)
        ax.fill_between(grid, dens, color=color, alpha=0.25)
        ax.axvline(np.mean(data), color=color, ls="--", lw=1)

    ax.set_title(f"Overall distribution of PC1..PC{k} scores (standardized per PC)\n"
                 f"covers {total_var:.1f}% cumulative variance")
    ax.set_xlabel("Standardized PC score (z)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.show()
    fig.canvas.draw_idle()


def run_violin(patches_real, gh_r, gw_r, proc_real,
               patches_synth, gh_s, gw_s, proc_synth, args,
               output=None, show=True, heatmap_out=None):
    """PCA + cumulative-variance violin figure from precomputed patch embeddings.
    Model-agnostic: used by the CLI below (dinov2-small) and by polar_pca_eval.py
    (any chosen foundation model, on polar-transformed images).

    output : output PNG path (defaults to args.output).
    show   : True (single run) opens the interactive click window; False
             (multi-model batch) only saves the static PNG and closes the figure."""
    X = np.vstack((patches_real, patches_synth))
    labels = np.concatenate([np.zeros(patches_real.shape[0]), np.ones(patches_synth.shape[0])])
    real_mask = labels == 0
    synth_mask = labels == 1
    n_real = patches_real.shape[0]
    grids = [(gh_r, gw_r), (gh_s, gw_s)]
    proc_imgs = [proc_real, proc_synth]
    print(f"Feature matrix: {X.shape[0]} patches x {X.shape[1]} dims "
          f"(real={patches_real.shape[0]}, synth={patches_synth.shape[0]})")

    print("Fitting PCA basis on the REAL embeddings; projecting both groups onto it...")
    pca = PCA(svd_solver="full", random_state=42).fit(patches_real)   # basis from REAL only
    scores = pca.transform(X)                 # (N, D) = (X - real_mean) @ components^T
    evr = pca.explained_variance_ratio_       # explained variance OF THE REAL data
    cum = np.cumsum(evr)

    # PCs up to (and including) the first one crossing the variance threshold.
    k = int(np.searchsorted(cum, args.var_threshold) + 1)
    k = min(k, len(cum))
    if args.max_pcs:
        k = min(k, args.max_pcs)
    k = max(k, 3)
    print(f"Showing PC1..PC{k} (cumulative variance reaches "
          f"{cum[k-1]*100:.2f}%; threshold {args.var_threshold*100:.0f}%).")

    # ---- main figure ---------------------------------------------------- #
    fig_w = max(12, k * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))
    fig.canvas.manager.set_window_title("PCA cumulative-variance violins")

    cum_pct = cum[:k] * 100
    half_height = args.violin_height
    half_width = 0.42

    # Cumulative-variance trend line through the violin centers.
    ax.plot(np.arange(1, k + 1), cum_pct, color="#444441", lw=1, alpha=0.5, zorder=1)
    ax.scatter(np.arange(1, k + 1), cum_pct, s=10, color="#444441", alpha=0.6, zorder=3)

    for pc in range(k):
        x_center = pc + 1
        draw_split_violin(ax, x_center, cum_pct[pc],
                          scores[real_mask, pc], scores[synth_mask, pc],
                          half_height, half_width)
        # Per-PC explained variance annotation above each violin.
        ax.text(x_center, cum_pct[pc] + half_height + 0.4, f"{evr[pc]*100:.1f}%",
                rotation=90, ha="center", va="bottom", fontsize=6, color="#5F5E5A")

    # Threshold reference line.
    ax.axhline(args.var_threshold * 100, color="gray", ls="--", lw=1)
    ax.text(0.6, args.var_threshold * 100 + 0.2,
            f"{args.var_threshold*100:.0f}% cumulative", fontsize=8, color="gray")

    # ---- overall violin (all PCs pooled, standardized per PC) ----------- #
    # z-score each PC's scores so every component contributes comparably,
    # then pool across PC1..PCk into one big split violin at the far right.
    z = scores[:, :k]
    zstd = z.std(axis=0)
    zstd[zstd < 1e-12] = 1.0
    z = (z - z.mean(axis=0)) / zstd
    overall_real = z[real_mask].ravel()
    overall_synth = z[synth_mask].ravel()

    y_bottom = cum_pct[0] - (half_height + 5)
    y_top = 101
    ax.set_ylim(y_bottom, y_top)

    x_overall = k + 2
    overall_h = (y_top - y_bottom) * 0.42   # spans most of the axis, like the reference
    overall_w = 0.9
    y_mid = 0.5 * (y_bottom + y_top)
    ax.axvline(k + 1, color="#cccccc", lw=0.8, ls=":")  # visual gap/separator
    draw_split_violin(ax, x_overall, y_mid, overall_real, overall_synth,
                      overall_h, overall_w)
    ax.text(x_overall, y_top - 0.4, f"Σ {cum_pct[k-1]:.1f}%",
            ha="center", va="top", fontsize=8, color="#5F5E5A")

    ax.set_xlim(0.3, x_overall + 1)
    ax.set_xticks(list(range(1, k + 1)) + [x_overall])
    ax.set_xticklabels([f"PC{i}" for i in range(1, k + 1)] + ["All PCs (z)"],
                       rotation=90, fontsize=max(5, min(9, int(400 / k))))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Distribution of PC scores along the cumulative-variance curve\n"
                 "(left = real, right = synthetic; PCA basis: real only, synth projected) — "
                 "click a violin to open that PC; far right = all PCs pooled",
                 fontsize=11)

    legend_handles = [Patch(facecolor=COLOR_REAL, label="Real"),
                      Patch(facecolor=COLOR_SYNTH, label="Synthetic"),
                      Line2D([0], [0], color="#444441", lw=1, label="Cumulative variance")]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = output or args.output
    _ensure_dir(out)
    fig.savefig(out, dpi=200)
    print(f"Saved static figure to {out}")

    # ---- variant B: optional static export of per-PC heatmaps ----------- #
    if args.heatmap_pcs:
        pcs = [p for p in parse_pc_list(args.heatmap_pcs) if 0 <= p < len(evr)]
        if pcs:
            save_pc_heatmaps(pcs, scores, n_real, grids, proc_imgs, evr,
                             heatmap_out or args.heatmap_out)

    # In multi-model batch mode we only save the static PNG (no blocking window).
    if not show:
        plt.close(fig)
        return

    # ---- click handler: open a PC in a separate window ------------------ #
    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.button != 1:
            return
        xr = int(round(event.xdata))
        if xr == x_overall and abs(event.ydata - y_mid) <= overall_h * 1.1:
            print("Opening overall PC distribution in a separate window...")
            make_overall_detail_window(overall_real, overall_synth, k, cum_pct[k - 1])
            return
        pc = xr - 1
        if 0 <= pc < k and abs(event.ydata - cum_pct[pc]) <= half_height * 1.5:
            print(f"Opening PC{pc + 1} (distribution + heatmap) in a separate window...")
            make_detail_window(pc, scores, real_mask, synth_mask, evr, cum,
                               grids, proc_imgs, n_real)

    fig.canvas.mpl_connect("button_press_event", on_click)

    backend = plt.get_backend().lower()
    if backend == "agg":
        print("\nNote: backend is non-interactive ('agg'); clicking won't work. "
              "Run with a GUI backend (TkAgg/QtAgg) for the click-to-open feature.")
    else:
        print("\nMain window is open. Click any violin to open that PC's distribution "
              "and patch heatmap in a new window. Close the window to exit.")
    plt.show()


def _with_suffix(name, suffix):
    """'bar.png' + 'dinov2-large' -> 'bar_dinov2-large.png'."""
    base, ext = os.path.splitext(name)
    return f"{base}_{suffix}{ext}"


def _lighten(hex_color, amount=0.55):
    """Blend a hex color toward white (amount in [0,1]) -> synth = lighter tone."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# Distinct base hues per model; real = base (dark tone), synth = lightened tone.
MODEL_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                 "#ff7f0e", "#8c564b", "#17becf", "#bcbd22"]


def _fit_model_scores(patches_real, patches_synth, args):
    """PCA on the REAL embeddings, project both; return (scores, evr, cum, k, n_real)."""
    X = np.vstack((patches_real, patches_synth))
    pca = PCA(svd_solver="full", random_state=42).fit(patches_real)
    scores = pca.transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, args.var_threshold) + 1)
    k = min(k, len(cum))
    if args.max_pcs:
        k = min(k, args.max_pcs)
    k = max(k, 3)
    return scores, evr, cum, k, patches_real.shape[0]


def run_violin_combined(per_model, args, output):
    """ONE figure overlaying ALL models: the same cumulative-variance split-violin
    layout as run_violin, but WITHOUT the 'All PCs (z)' pooled violin. Each model
    gets its own hue (real = dark tone on the left, synth = light tone on the
    right) and its PCs are connected by that model's cumulative-variance curve, so
    the real-vs-synthetic comparison is kept, just per foundation model."""
    results = []
    for md in per_model:
        scores, evr, cum, k, n_real = _fit_model_scores(md["pr"], md["ps"], args)
        results.append({"name": md["name"], "scores": scores, "evr": evr,
                        "cum": cum, "k": k, "n_real": n_real})
    max_k = max(r["k"] for r in results)
    M = len(results)

    fig_w = max(12, max_k * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))
    fig.canvas.manager.set_window_title("PCA cumulative-variance violins - all models")

    # Horizontal dodge so models' violins at the same PC do not sit on top of each other.
    offsets = np.linspace(-0.3, 0.3, M) if M > 1 else np.array([0.0])
    half_width = min(0.4, 0.8 / M)
    half_height = args.violin_height

    legend_handles = []
    for mi, r in enumerate(results):
        base = MODEL_PALETTE[mi % len(MODEL_PALETTE)]
        c_real, c_synth = base, _lighten(base, 0.55)
        cum_pct = r["cum"][:r["k"]] * 100
        xs = np.arange(1, r["k"] + 1) + offsets[mi]
        real_mask = np.zeros(r["scores"].shape[0], bool)
        real_mask[:r["n_real"]] = True

        ax.plot(xs, cum_pct, color=base, lw=1.3, alpha=0.9, zorder=2)
        ax.scatter(xs, cum_pct, s=10, color=base, alpha=0.9, zorder=3)
        for pc in range(r["k"]):
            draw_split_violin(ax, xs[pc], cum_pct[pc],
                              r["scores"][real_mask, pc], r["scores"][~real_mask, pc],
                              half_height, half_width,
                              color_real=c_real, color_synth=c_synth)
        legend_handles.append(Patch(facecolor=base, label=r["name"]))

    ax.axhline(args.var_threshold * 100, color="gray", ls="--", lw=1)
    ax.text(0.6, args.var_threshold * 100 + 0.2,
            f"{args.var_threshold*100:.0f}% cumulative", fontsize=8, color="gray")

    ax.set_xlim(0.3, max_k + 1)
    ax.set_xticks(list(range(1, max_k + 1)))
    ax.set_xticklabels([f"PC{i}" for i in range(1, max_k + 1)],
                       rotation=90, fontsize=max(5, min(9, int(400 / max_k))))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PC-score distributions along the cumulative-variance curve - all models\n"
                 "(per model: dark/left half = real, light/right half = synthetic; "
                 "PCA basis = real only, synthetic projected)", fontsize=11)

    legend_handles.append(Line2D([0], [0], color="#888888", lw=0,
                                 marker="s", markerfacecolor="#888888",
                                 label="dark = real / light = synth"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, title="Modell")

    fig.tight_layout()
    _ensure_dir(output)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved combined all-models figure to {output}")


def run_violin_multi(per_model, args):
    """Cumulative-variance violin analysis for ONE OR MORE foundation models.

    per_model : list of dicts with keys name, pr, ghr, gwr, disp_r, ps, ghs, gws,
        disp_s. A single model reproduces the original interactive figure; with
        several, each model gets its own static PNG (..._<name>.png) plus ONE
        combined figure (..._all_models.png) overlaying all of them."""
    image_dir = getattr(args, "image_dir", None) or os.path.dirname(args.output) or "."
    base = os.path.basename(args.output)
    hbase = os.path.basename(args.heatmap_out)

    if len(per_model) == 1:
        md = per_model[0]
        run_violin(md["pr"], md["ghr"], md["gwr"], md["disp_r"],
                   md["ps"], md["ghs"], md["gws"], md["disp_s"], args,
                   output=os.path.join(image_dir, base), show=True,
                   heatmap_out=os.path.join(image_dir, hbase))
        return

    for md in per_model:
        out = os.path.join(image_dir, _with_suffix(base, md["name"]))
        print(f"\n[{md['name']}] cumulative-variance violin -> {out}")
        run_violin(md["pr"], md["ghr"], md["gwr"], md["disp_r"],
                   md["ps"], md["ghs"], md["gws"], md["disp_s"], args,
                   output=out, show=False,
                   heatmap_out=os.path.join(image_dir, _with_suffix(hbase, md["name"])))

    combined = os.path.join(image_dir, _with_suffix(base, "all_models"))
    print("\n[all models] combined comparison figure ...")
    run_violin_combined(per_model, args, combined)


def main(args):
    print("Loading DINOv2 model...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()

    if not os.path.exists(args.real_image) or not os.path.exists(args.synthetic_image):
        print("Error: Could not find image files.")
        print(f"Please ensure both '{args.real_image}' and '{args.synthetic_image}' exist.")
        return

    print("Loading images and extracting DINOv2 patch embeddings...")
    patches_real, gh_r, gw_r, proc_real = extract(model, processor, load_image(args.real_image))
    patches_synth, gh_s, gw_s, proc_synth = extract(model, processor, load_image(args.synthetic_image))
    run_violin(patches_real, gh_r, gw_r, proc_real,
               patches_synth, gh_s, gw_s, proc_synth, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive cumulative-variance violin plot of DINOv2 PCA scores.")
    parser.add_argument("--real_image", type=str, default="real.png",
                        help="Path to the real CT scan image (PNG)")
    parser.add_argument("--synthetic_image", type=str, default="fake.png",
                        help="Path to the synthetic CT scan image (PNG)")
    parser.add_argument("--var_threshold", type=float, default=0.80,
                        help="Show PCs until cumulative variance exceeds this (0-1)")
    parser.add_argument("--max_pcs", type=int, default=None,
                        help="Optional hard cap on the number of PCs shown")
    parser.add_argument("--violin_height", type=float, default=3.0,
                        help="Half-height of each violin in cumulative-variance %% units")
    parser.add_argument("--output", type=str, default="pca_cumulative_violin.png",
                        help="Path for the saved static figure")
    parser.add_argument("--heatmap_pcs", type=str, default=None,
                        help="Variant B: also save per-PC patch heatmaps for these "
                             "PCs, e.g. '1-6' or '1,3,5' (1-based)")
    parser.add_argument("--heatmap_out", type=str, default="pca_pc_heatmaps.png",
                        help="Output path for the per-PC heatmap figure")
    args = parser.parse_args()
    main(args)
