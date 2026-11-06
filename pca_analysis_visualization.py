"""
PCA analysis visualization for DINOv2 latent space.

This script makes the dimensionality reduction behind PCA *tangible*: DINOv2
produces 384-dimensional patch embeddings (ViT-S/14), and PCA projects them to
a few components. Each plot answers a concrete question about that reduction.

The feature matrix X is the set of ALL patch tokens of both images
(n_patches_real + n_patches_synth, 384). Every patch is one sample; the color
encodes whether it came from the real or the synthetic scan.

Plots produced (saved into --output_dir):
  1. scree_plot.png        -> How much variance does each PC keep? (How many dims for 90/95%?)
  2. reconstruction_error.png -> MSE when reconstructing X from k components.
  3. scatter_2d.png        -> PC1 vs PC2 with marginal histograms, colored real/synth.
  4. scatter_3d.png        -> First 3 PCs in 3D.
  5. pairplot.png          -> Scatter matrix of the first N PCs (KDE on the diagonal).
  6. component_kde.png     -> Per-component score distributions real vs synth (distribution shift).
  7. patch_rgb.png         -> DINOv2 spatial map: first 3 PCs of foreground patches -> RGB overlay.

Model loading is identical to cluster_ct_scans.py (facebook/dinov2-small, patch size 14).
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Consistent colors for the two groups across all plots.
COLOR_REAL = "#2c7fb8"   # blue
COLOR_SYNTH = "#d95f0e"  # orange


def load_image(image_path):
    """Load image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def extract_patch_embeddings(model, processor, image, patch_size=14):
    """Run DINOv2 and return (patch_embeddings [N, D], grid_h, grid_w, cls [D])."""
    inputs = processor(images=image, return_tensors="pt")
    h, w = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
    grid_h, grid_w = h // patch_size, w // patch_size

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (batch, sequence_length, hidden_size).
    # Token 0 is the CLS token, the rest are the patch tokens.
    patch_embeddings = outputs.last_hidden_state[0, 1:, :].numpy()
    cls = outputs.last_hidden_state[0, 0, :].numpy()

    if grid_h * grid_w != patch_embeddings.shape[0]:
        print(f"Warning: grid {grid_h}x{grid_w} != token count {patch_embeddings.shape[0]}")

    return patch_embeddings, grid_h, grid_w, cls


# --------------------------------------------------------------------------- #
# Plot 1: Scree plot (explained variance per PC + cumulative)
# --------------------------------------------------------------------------- #
def plot_scree(pca, output_dir, max_components=40):
    evr = pca.explained_variance_ratio_
    n = min(max_components, len(evr))
    cum = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(1, n + 1)
    ax1.bar(x, evr[:n] * 100, color=COLOR_REAL, alpha=0.7, label="Explained variance per PC")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance per PC (%)", color=COLOR_REAL)
    ax1.tick_params(axis="y", labelcolor=COLOR_REAL)

    ax2 = ax1.twinx()
    ax2.plot(x, cum[:n] * 100, color=COLOR_SYNTH, marker="o", ms=4, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance (%)", color=COLOR_SYNTH)
    ax2.tick_params(axis="y", labelcolor=COLOR_SYNTH)
    ax2.set_ylim(0, 101)

    # Mark how many PCs are needed for 90% / 95%.
    for thr, style in [(0.90, "--"), (0.95, ":")]:
        k = int(np.searchsorted(cum, thr) + 1)
        ax2.axhline(thr * 100, color="gray", ls=style, lw=1)
        if k <= n:
            ax2.axvline(k, color="gray", ls=style, lw=1)
            ax2.annotate(f"{int(thr*100)}% @ {k} PCs", xy=(k, thr * 100),
                         xytext=(k + 0.5, thr * 100 - 8), fontsize=9, color="gray")

    plt.title("Scree plot — variance kept by each principal component")
    fig.tight_layout()
    path = os.path.join(output_dir, "scree_plot.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 2: Reconstruction error vs k
# --------------------------------------------------------------------------- #
def plot_reconstruction_error(X, pca, output_dir, max_k=None):
    # Project to all components once, then truncate to k and reconstruct.
    scores = pca.transform(X)                 # (n, n_comp)
    components = pca.components_               # (n_comp, D)
    mean = pca.mean_                           # (D,)
    n_comp = scores.shape[1]
    if max_k is None:
        max_k = n_comp
    ks = np.arange(1, min(max_k, n_comp) + 1)

    errors = []
    var_total = np.var(X, axis=0).sum()
    for k in ks:
        X_recon = mean + scores[:, :k] @ components[:k]
        mse = np.mean((X - X_recon) ** 2)
        errors.append(mse)
    errors = np.array(errors)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, errors, color=COLOR_SYNTH, marker="o", ms=3)
    ax.set_xlabel("Number of components k")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction error vs. number of PCA components")
    ax.grid(alpha=0.3)

    # Annotate the typical low-dim targets used for visualization.
    for k in [2, 3]:
        if k <= ks[-1]:
            ax.axvline(k, color="gray", ls="--", lw=1)
            ax.annotate(f"k={k}\nMSE={errors[k-1]:.3g}", xy=(k, errors[k - 1]),
                        xytext=(k + 1, errors[k - 1]), fontsize=9, color="gray")

    fig.tight_layout()
    path = os.path.join(output_dir, "reconstruction_error.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 3: 2D scatter with marginal histograms
# --------------------------------------------------------------------------- #
def plot_scatter_2d(scores, labels, evr, output_dir):
    real = labels == 0
    synth = labels == 1

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax)

    ax.scatter(scores[real, 0], scores[real, 1], s=12, alpha=0.6, c=COLOR_REAL, label="Real patches")
    ax.scatter(scores[synth, 0], scores[synth, 1], s=12, alpha=0.6, c=COLOR_SYNTH, label="Synthetic patches")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.legend(loc="best")

    for mask, color in [(real, COLOR_REAL), (synth, COLOR_SYNTH)]:
        ax_top.hist(scores[mask, 0], bins=40, color=color, alpha=0.5, density=True)
        ax_right.hist(scores[mask, 1], bins=40, color=color, alpha=0.5,
                      density=True, orientation="horizontal")
    ax_top.axis("off")
    ax_right.axis("off")

    fig.suptitle("PCA 2D scatter (PC1 vs PC2) with marginal distributions", y=0.92)
    path = os.path.join(output_dir, "scatter_2d.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 4: 3D scatter of first 3 PCs
# --------------------------------------------------------------------------- #
def plot_scatter_3d(scores, labels, evr, output_dir):
    if scores.shape[1] < 3:
        return
    real = labels == 0
    synth = labels == 1

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores[real, 0], scores[real, 1], scores[real, 2],
               s=10, alpha=0.5, c=COLOR_REAL, label="Real patches")
    ax.scatter(scores[synth, 0], scores[synth, 1], scores[synth, 2],
               s=10, alpha=0.5, c=COLOR_SYNTH, label="Synthetic patches")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)")
    ax.set_title("PCA 3D scatter (first 3 components)")
    ax.legend()

    path = os.path.join(output_dir, "scatter_3d.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 5: Pairplot / scatter matrix of first N PCs
# --------------------------------------------------------------------------- #
def plot_pairplot(scores, labels, evr, output_dir, n_pcs=4):
    n = min(n_pcs, scores.shape[1])
    real = labels == 0
    synth = labels == 1

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # KDE on the diagonal for each group.
                for mask, color in [(real, COLOR_REAL), (synth, COLOR_SYNTH)]:
                    data = scores[mask, i]
                    if data.size > 2 and np.std(data) > 1e-8:
                        kde = gaussian_kde(data)
                        grid = np.linspace(data.min(), data.max(), 200)
                        ax.plot(grid, kde(grid), color=color, lw=1.5)
                        ax.fill_between(grid, kde(grid), color=color, alpha=0.25)
            else:
                ax.scatter(scores[real, j], scores[real, i], s=4, alpha=0.4, c=COLOR_REAL)
                ax.scatter(scores[synth, j], scores[synth, i], s=4, alpha=0.4, c=COLOR_SYNTH)
            if i == n - 1:
                ax.set_xlabel(f"PC{j+1}")
            if j == 0:
                ax.set_ylabel(f"PC{i+1}")
            ax.tick_params(labelsize=7)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_REAL, markersize=8, label="Real"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SYNTH, markersize=8, label="Synthetic"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")
    fig.suptitle(f"Scatter matrix of the first {n} principal components", y=1.0)
    fig.tight_layout()
    path = os.path.join(output_dir, "pairplot.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 6: Per-component KDE (distribution shift real vs synth)
# --------------------------------------------------------------------------- #
def plot_component_kde(scores, labels, evr, output_dir, n_pcs=6):
    n = min(n_pcs, scores.shape[1])
    real = labels == 0
    synth = labels == 1

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for c in range(n):
        ax = axes[c]
        for mask, color, name in [(real, COLOR_REAL, "Real"), (synth, COLOR_SYNTH, "Synthetic")]:
            data = scores[mask, c]
            if data.size > 2 and np.std(data) > 1e-8:
                kde = gaussian_kde(data)
                grid = np.linspace(scores[:, c].min(), scores[:, c].max(), 300)
                ax.plot(grid, kde(grid), color=color, lw=1.8, label=name)
                ax.fill_between(grid, kde(grid), color=color, alpha=0.25)
        ax.set_title(f"PC{c+1} ({evr[c]*100:.1f}% var)")
        ax.set_yticks([])
        if c == 0:
            ax.legend(fontsize=8)

    for c in range(n, len(axes)):
        axes[c].axis("off")

    fig.suptitle("Distribution of PC scores: real vs. synthetic", y=1.0)
    fig.tight_layout()
    path = os.path.join(output_dir, "component_kde.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Plot 7: DINOv2 spatial patch-token RGB map (the signature DINOv2 viz)
# --------------------------------------------------------------------------- #
def plot_patch_rgb(patches_real, grid_real, patches_synth, grid_synth,
                   img_real, img_synth, output_dir, fg_threshold=0.5, invert_fg=False):
    """
    Standard DINOv2 recipe:
      1. PCA over all patches; use PC1 to separate foreground from background.
      2. A second PCA on the foreground patches, mapped to RGB.
    Background patches are shown black. Same structures get consistent colors
    across the two images, which makes failure modes visible.
    """
    grid_h_r, grid_w_r = grid_real
    grid_h_s, grid_w_s = grid_synth
    n_real = patches_real.shape[0]

    X = np.vstack((patches_real, patches_synth))

    # Step 1: PC1 over all patches -> foreground mask.
    pca1 = PCA(n_components=1, random_state=42)
    pc1 = pca1.fit_transform(X)[:, 0]
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    fg = pc1_norm > fg_threshold
    if invert_fg:
        fg = ~fg

    # Step 2: PCA(3) on foreground patches -> RGB.
    rgb = np.zeros((X.shape[0], 3))
    if fg.sum() >= 3:
        pca3 = PCA(n_components=3, random_state=42)
        fg_rgb = pca3.fit_transform(X[fg])
        fg_rgb = (fg_rgb - fg_rgb.min(axis=0)) / (fg_rgb.max(axis=0) - fg_rgb.min(axis=0) + 1e-8)
        rgb[fg] = fg_rgb

    rgb_real = rgb[:n_real].reshape(grid_h_r, grid_w_r, 3)
    rgb_synth = rgb[n_real:].reshape(grid_h_s, grid_w_s, 3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes[0, 0].imshow(img_real)
    axes[0, 0].set_title("Real scan — input")
    axes[0, 1].imshow(rgb_real)
    axes[0, 1].set_title("Real scan — PCA patch RGB (foreground)")
    axes[1, 0].imshow(img_synth)
    axes[1, 0].set_title("Synthetic scan — input")
    axes[1, 1].imshow(rgb_synth)
    axes[1, 1].set_title("Synthetic scan — PCA patch RGB (foreground)")
    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("DINOv2 spatial PCA: first 3 PCs of foreground patches -> RGB", y=0.98)
    fig.tight_layout()
    path = os.path.join(output_dir, "patch_rgb.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


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

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading images...")
    img_real = load_image(args.real_image)
    img_synth = load_image(args.synthetic_image)

    print("Extracting DINOv2 patch embeddings...")
    patches_real, gh_r, gw_r, cls_real = extract_patch_embeddings(model, processor, img_real)
    patches_synth, gh_s, gw_s, cls_synth = extract_patch_embeddings(model, processor, img_synth)

    print("\n--- Global Image Distance (CLS tokens) ---")
    print(f"Cosine Distance:    {cosine(cls_real, cls_synth):.4f}")
    print(f"Euclidean Distance: {euclidean(cls_real, cls_synth):.4f}")

    # Feature matrix: every patch token is a sample (real=0, synthetic=1).
    X = np.vstack((patches_real, patches_synth))
    labels = np.concatenate([np.zeros(patches_real.shape[0]), np.ones(patches_synth.shape[0])])
    print(f"\nFeature matrix: {X.shape[0]} patches x {X.shape[1]} dims "
          f"(real={patches_real.shape[0]}, synth={patches_synth.shape[0]})")

    if args.standardize:
        print("Standardizing features (zero mean, unit variance per dimension)...")
        X = StandardScaler().fit_transform(X)

    # One PCA with many components feeds every analysis plot.
    n_comp = min(args.n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"PCA fitted with {n_comp} components. "
          f"PC1-3 keep {cum[2]*100:.1f}% of variance; "
          f"{int(np.searchsorted(cum, 0.90) + 1)} PCs reach 90%.")

    print("\nGenerating plots...")
    plot_scree(pca, args.output_dir)
    plot_reconstruction_error(X, pca, args.output_dir, max_k=args.n_components)
    plot_scatter_2d(scores, labels, evr, args.output_dir)
    plot_scatter_3d(scores, labels, evr, args.output_dir)
    plot_pairplot(scores, labels, evr, args.output_dir, n_pcs=args.pairplot_pcs)
    plot_component_kde(scores, labels, evr, args.output_dir, n_pcs=args.kde_pcs)
    plot_patch_rgb(patches_real, (gh_r, gw_r), patches_synth, (gh_s, gw_s),
                   img_real, img_synth, args.output_dir,
                   fg_threshold=args.fg_threshold, invert_fg=args.invert_fg)

    print(f"\nDone. All plots saved in '{args.output_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the PCA analysis behind DINOv2 patch embeddings.")
    parser.add_argument("--real_image", type=str, default="real.png",
                        help="Path to the real CT scan image (PNG)")
    parser.add_argument("--synthetic_image", type=str, default="fake.png",
                        help="Path to the synthetic CT scan image (PNG)")
    parser.add_argument("--output_dir", type=str, default="pca_plots",
                        help="Directory to save the plots into")
    parser.add_argument("--n_components", type=int, default=50,
                        help="Number of PCA components to fit (capped by data size)")
    parser.add_argument("--pairplot_pcs", type=int, default=4,
                        help="Number of PCs shown in the scatter matrix")
    parser.add_argument("--kde_pcs", type=int, default=6,
                        help="Number of PCs shown in the per-component KDE plot")
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize features before PCA (default: only centering)")
    parser.add_argument("--fg_threshold", type=float, default=0.5,
                        help="PC1 threshold (0-1) for foreground/background split in patch_rgb")
    parser.add_argument("--invert_fg", action="store_true",
                        help="Invert the foreground mask if background gets colored instead")
    args = parser.parse_args()
    main(args)
