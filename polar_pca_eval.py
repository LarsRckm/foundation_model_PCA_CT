"""
Polar PCA evaluation (new entry point).

Pipeline:  user-chosen real + synthetic image (each in a user-stated format)
  -> cart->polar transform for cartesian inputs (generator's polar_transform.py,
     imported read-only; polar inputs are used directly)
  -> user-chosen foundation model -> patch embeddings of both images
  -> EITHER the cumulative-variance violin analysis (pca_cumulative_violin)
     OR the Excel + interactive-3D analysis (pca_excel_interactive).

The two analysis scripts are reused unchanged (their run_violin / run_excel
functions); only the front end (polar transform + model choice + embedding) is
new, in polar_embed.py.

Examples:
  # cumulative-variance violins, DINOv2-large, real cartesian + synth polar
  python polar_pca_eval.py --analysis violin --model dinov2-large \
      --real_image real_cart.png      --real_format cart \
      --synthetic_image fake_polar.png --synth_format polar

  # Excel report + interactive 3D HTML, DINOv3, both cartesian
  python polar_pca_eval.py --analysis excel --model dinov3-large \
      --real_image real.png --real_format cart \
      --synthetic_image fake.png --synth_format cart
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import polar_embed as pe                     # noqa: E402
import pca_cumulative_violin as violin       # noqa: E402  (reused: run_violin)
import pca_excel_interactive as excel        # noqa: E402  (reused: run_excel)


def _region_counts(reg):
    """Readable patch count per region for the console."""
    names = {1: "mandrel", 2: "layers", 3: "housing", 0: "pad"}
    return {names[k]: int((reg == k).sum()) for k in (1, 2, 3, 0)}


def main(args):
    for path, which in ((args.real_image, "real"), (args.synthetic_image, "synthetic")):
        if not os.path.exists(path):
            raise SystemExit(f"{which} image not found: {path}")

    # Polar transform fns are only needed when at least one input is cartesian.
    fns = None
    if "cart" in (args.real_format, args.synth_format):
        try:
            fns = pe.load_polar_fns(args.polar_module)
        except Exception as exc:
            raise SystemExit(
                f"Could not import polar_transform from --polar_module '{args.polar_module}':\n"
                f"  {exc}\n"
                "Cartesian inputs need the generator's polar_transform.py (OpenCV). Point "
                "--polar_module at the folder that contains it, or pass already-polar images "
                "with --real_format polar / --synth_format polar.")

    print(f"Loading foundation model '{args.model}' ...")
    model, proc, patch, device = pe.load_model(args.model, args.device)
    H, W = pe.frame_size(args.n_r, args.n_theta, patch)
    gh, gw = H // patch, W // patch
    print(f"  {args.model} | patch {patch} | polar {args.n_r}x{args.n_theta} "
          f"-> frame {H}x{W} -> grid {gh}x{gw} = {gh * gw} patch tokens | {device}")

    print(f"Embedding REAL image ({args.real_format}) ...")
    pr, ghr, gwr, _mr, disp_r = pe.embed_image(
        args.real_image, args.real_format, model, proc, patch, device,
        args.n_r, args.n_theta, fns, args.pad_thresh)
    print(f"  real:  {pr.shape[0]} patches x {pr.shape[1]} dims  (grid {ghr}x{gwr})")

    print(f"Embedding SYNTHETIC image ({args.synth_format}) ...")
    ps, ghs, gws, _ms, disp_s = pe.embed_image(
        args.synthetic_image, args.synth_format, model, proc, patch, device,
        args.n_r, args.n_theta, fns, args.pad_thresh)
    print(f"  synth: {ps.shape[0]} patches x {ps.shape[1]} dims  (grid {ghs}x{gws})")

    if args.analysis == "violin":
        print("\nRunning cumulative-variance violin analysis ...")
        violin.run_violin(pr, ghr, gwr, disp_r, ps, ghs, gws, disp_s, args)
    else:
        print("\nRunning Excel + interactive-3D analysis ...")

        # Classify every patch into mandrel / layers / housing with the generator's
        # detect_regions_per_angle (read-only) so the HTML can color points by region
        # and draw the two transitions on the polar image.
        regions = bounds = None
        try:
            region_fn = pe.load_region_fn(args.polar_module)
            reg_r, bnd_r = pe.patch_regions(
                args.real_image, args.real_format, args.n_r, args.n_theta,
                H, W, patch, fns, region_fn, args.pad_thresh)
            reg_s, bnd_s = pe.patch_regions(
                args.synthetic_image, args.synth_format, args.n_r, args.n_theta,
                H, W, patch, fns, region_fn, args.pad_thresh)
            regions, bounds = [reg_r, reg_s], [bnd_r, bnd_s]
            print(f"  regions real:  {_region_counts(reg_r)}")
            print(f"  regions synth: {_region_counts(reg_s)}")
        except Exception as exc:
            print(f"  [warn] region classification unavailable ({exc}); "
                  "HTML will have no region overlay.")

        excel.run_excel(pr, ghr, gwr, disp_r, ps, ghs, gws, disp_s,
                        os.path.basename(args.real_image),
                        os.path.basename(args.synthetic_image), args,
                        regions=regions, bounds=bounds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Polar cart->polar + foundation-model patch embeddings -> "
                    "cumulative-variance violin OR Excel+interactive-3D PCA analysis.")

    # --- images + their formats (user-specified) ---
    p.add_argument("--real_image", required=True, help="Path to the REAL CT image")
    p.add_argument("--synthetic_image", required=True, help="Path to the SYNTHETIC CT image")
    p.add_argument("--real_format", choices=["cart", "polar"], default="cart",
                   help="Real image cartesian (cart->polar) or already polar")
    p.add_argument("--synth_format", choices=["cart", "polar"], default="polar",
                   help="Synthetic image cartesian (cart->polar) or already polar")

    # --- foundation model ---
    p.add_argument("--model", default="dinov2-small", choices=sorted(pe.MODELS),
                   help="Foundation model for the patch embeddings")

    # --- which analysis ---
    p.add_argument("--analysis", choices=["violin", "excel"], default="violin",
                   help="violin = pca_cumulative_violin ; excel = pca_excel_interactive")

    # --- polar transform params (match create_embeddings.py) ---
    p.add_argument("--n_r", type=int, default=512, help="Polar radial samples before reframing")
    p.add_argument("--n_theta", type=int, default=1024, help="Polar angular samples before reframing")
    p.add_argument("--pad_thresh", type=int, default=5, help="Grey <= this = padding (native polar)")
    p.add_argument("--polar_module", default=pe.DEFAULT_POLAR,
                   help="Folder with the generator's polar_transform.py (for cartesian inputs)")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")

    # --- violin outputs (forwarded to run_violin) ---
    p.add_argument("--var_threshold", type=float, default=0.70,
                   help="[violin] show PCs until cumulative variance exceeds this (0-1)")
    p.add_argument("--max_pcs", type=int, default=None, help="[violin] hard cap on PCs shown")
    p.add_argument("--violin_height", type=float, default=3.0, help="[violin] half-height per violin")
    p.add_argument("--output", default=os.path.join(HERE, "polar_pca_cumulative_violin.png"),
                   help="[violin] saved static figure")
    p.add_argument("--heatmap_pcs", default=None,
                   help="[violin] also save per-PC patch heatmaps, e.g. '1-6' or '1,3,5'")
    p.add_argument("--heatmap_out", default=os.path.join(HERE, "polar_pca_pc_heatmaps.png"),
                   help="[violin] per-PC heatmap figure path")

    # --- excel outputs (forwarded to run_excel) ---
    p.add_argument("--excel_out", default=os.path.join(HERE, "polar_pca_variance.xlsx"),
                   help="[excel] variance report path")
    p.add_argument("--html_out", default=os.path.join(HERE, "polar_pca_interactive_3d.html"),
                   help="[excel] interactive 3D HTML path")

    main(p.parse_args())
