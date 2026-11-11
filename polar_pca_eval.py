"""
Polar PCA evaluation (entry point).

Pipeline:  user-chosen real + synthetic image (each in a user-stated format)
  -> cart->polar transform for cartesian inputs (generator's polar_transform.py,
     imported read-only; polar inputs are used directly)
  -> ONE OR MORE user-chosen foundation models -> patch embeddings of both images
     (each model with its own patch size; embedded once, then reused)
  -> the cumulative-variance violin analysis (pca_cumulative_violin) AND/OR
     the Excel + interactive-3D analysis (pca_excel_interactive).

Multiple models: each model yields its own interactive HTML and its own violin
figure; a single Excel workbook collects every model (two sheets each), and an
extra violin figure overlays all models for direct comparison. Results go into
results/html, results/excel and results/images (created automatically).

Gated models (e.g. dinov3-large, patch size 16) need a Hugging Face token: pass
--hf_token, set the HF_TOKEN env var, or run `huggingface-cli login`, and request
access to the model repo on huggingface.co first.

Examples:
  # both analyses, three models at once, real cartesian + synth polar
  python polar_pca_eval.py \
      --models dinov2-small dinov2-large dinov3-large --hf_token hf_xxx \
      --real_image real_cart.png      --real_format cart \
      --synthetic_image fake_polar.png --synth_format polar

  # only the violin comparison, two models, both cartesian
  python polar_pca_eval.py --analysis violin --models dinov2-large dinov3-large \
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


def _resolve_models(args):
    """Model list from --models (preferred) or the single --model alias."""
    if args.models:
        models = list(args.models)
    elif args.model:
        models = [args.model]
    else:
        models = ["dinov2-small"]
    seen = []
    for m in models:                       # dedupe, keep order
        if m not in seen:
            seen.append(m)
    return seen


def main(args):
    for path, which in ((args.real_image, "real"), (args.synthetic_image, "synthetic")):
        if not os.path.exists(path):
            raise SystemExit(f"{which} image not found: {path}")

    models = _resolve_models(args)
    analyses = list(args.analysis)
    print(f"Models: {', '.join(models)}   |   analyses: {', '.join(analyses)}")

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

    # Region classifier (mandrel / layers / housing) is only needed for Excel/HTML.
    region_fn = None
    if "excel" in analyses:
        try:
            region_fn = pe.load_region_fn(args.polar_module)
        except Exception as exc:
            print(f"  [warn] region classifier unavailable ({exc}); "
                  "HTML will have no region overlay.")

    # Embed BOTH images with every requested model (each model only once), honoring
    # that model's own patch size, and collect everything for the analyses below.
    per_model = []
    for name in models:
        print(f"\n=== Foundation model '{name}' ===")
        model, proc, patch, device = pe.load_model(name, args.device, token=args.hf_token)
        H, W = pe.frame_size(args.n_r, args.n_theta, patch)
        gh, gw = H // patch, W // patch
        print(f"  {name} | patch {patch} | polar {args.n_r}x{args.n_theta} "
              f"-> frame {H}x{W} -> grid {gh}x{gw} = {gh * gw} patch tokens | {device}")

        print(f"  embedding REAL ({args.real_format}) ...")
        pr, ghr, gwr, _mr, disp_r = pe.embed_image(
            args.real_image, args.real_format, model, proc, patch, device,
            args.n_r, args.n_theta, fns, args.pad_thresh)
        print(f"    real:  {pr.shape[0]} patches x {pr.shape[1]} dims  (grid {ghr}x{gwr})")

        print(f"  embedding SYNTHETIC ({args.synth_format}) ...")
        ps, ghs, gws, _ms, disp_s = pe.embed_image(
            args.synthetic_image, args.synth_format, model, proc, patch, device,
            args.n_r, args.n_theta, fns, args.pad_thresh)
        print(f"    synth: {ps.shape[0]} patches x {ps.shape[1]} dims  (grid {ghs}x{gws})")

        regions = bounds = None
        if "excel" in analyses and region_fn is not None:
            try:
                reg_r, bnd_r = pe.patch_regions(
                    args.real_image, args.real_format, args.n_r, args.n_theta,
                    H, W, patch, fns, region_fn, args.pad_thresh)
                reg_s, bnd_s = pe.patch_regions(
                    args.synthetic_image, args.synth_format, args.n_r, args.n_theta,
                    H, W, patch, fns, region_fn, args.pad_thresh)
                regions, bounds = [reg_r, reg_s], [bnd_r, bnd_s]
                print(f"    regions real:  {_region_counts(reg_r)}")
                print(f"    regions synth: {_region_counts(reg_s)}")
            except Exception as exc:
                print(f"    [warn] region classification for {name} unavailable ({exc}).")

        per_model.append({
            "name": name,
            "pr": pr, "ghr": ghr, "gwr": gwr, "disp_r": disp_r,
            "ps": ps, "ghs": ghs, "gws": gws, "disp_s": disp_s,
            "regions": regions, "bounds": bounds,
        })

        # Free the model before the next one (dinov2-giant / dinov3-large are heavy).
        del model, proc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if "violin" in analyses:
        print("\n=== Cumulative-variance violin analysis ===")
        violin.run_violin_multi(per_model, args)

    if "excel" in analyses:
        print("\n=== Excel report + interactive-3D analysis ===")
        excel.run_excel_multi(per_model, args)


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

    # --- foundation model(s) ---
    p.add_argument("--models", nargs="+", choices=sorted(pe.MODELS), default=None,
                   help="One or more foundation models applied to BOTH images, e.g. "
                        "--models dinov2-small dinov2-large dinov3-large")
    p.add_argument("--model", choices=sorted(pe.MODELS), default=None,
                   help="(deprecated) single-model alias for --models")
    p.add_argument("--hf_token", default=None,
                   help="Hugging Face access token for GATED models (e.g. dinov3-large); "
                        "alternatively set the HF_TOKEN env var or run `huggingface-cli login`")

    # --- which analysis (one or both) ---
    p.add_argument("--analysis", nargs="+", choices=["violin", "excel"],
                   default=["violin", "excel"],
                   help="violin = pca_cumulative_violin ; excel = pca_excel_interactive "
                        "(default: both — each model is embedded only once)")

    # --- polar transform params (match create_embeddings.py) ---
    p.add_argument("--n_r", type=int, default=512, help="Polar radial samples before reframing")
    p.add_argument("--n_theta", type=int, default=1024, help="Polar angular samples before reframing")
    p.add_argument("--pad_thresh", type=int, default=5, help="Grey <= this = padding (native polar)")
    p.add_argument("--polar_module", default=pe.DEFAULT_POLAR,
                   help="Folder with the generator's polar_transform.py (for cartesian inputs)")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")

    # --- result folders (created automatically) ---
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"),
                   help="Parent folder for results; html/excel/images subfolders live here")
    p.add_argument("--html_dir", default=None,
                   help="Folder for the interactive HTML files (default: <results_dir>/html)")
    p.add_argument("--excel_dir", default=None,
                   help="Folder for the Excel report (default: <results_dir>/excel)")
    p.add_argument("--image_dir", default=None,
                   help="Folder for the PNG images (default: <results_dir>/images)")

    # --- violin outputs (forwarded to run_violin_multi) ---
    p.add_argument("--var_threshold", type=float, default=0.70,
                   help="[violin] show PCs until cumulative variance exceeds this (0-1)")
    p.add_argument("--max_pcs", type=int, default=None, help="[violin] hard cap on PCs shown")
    p.add_argument("--violin_height", type=float, default=3.0, help="[violin] half-height per violin")
    p.add_argument("--output", default="polar_pca_cumulative_violin.png",
                   help="[violin] base file name for the per-model figure(s) (in --image_dir); "
                        "with several models also a <base>_all_models.png comparison figure")
    p.add_argument("--heatmap_pcs", default=None,
                   help="[violin] also save per-PC patch heatmaps, e.g. '1-6' or '1,3,5'")
    p.add_argument("--heatmap_out", default="polar_pca_pc_heatmaps.png",
                   help="[violin] per-PC heatmap figure file name (in --image_dir)")

    # --- excel outputs (forwarded to run_excel_multi) ---
    p.add_argument("--excel_out", default="polar_pca_variance.xlsx",
                   help="[excel] Excel report file name (in --excel_dir); one workbook, "
                        "two sheets per model (<model>_real / <model>_synth)")
    p.add_argument("--html_out", default="polar_pca_interactive_3d.html",
                   help="[excel] interactive 3D HTML base file name (in --html_dir); "
                        "one file per model")

    args = p.parse_args()
    args.html_dir = args.html_dir or os.path.join(args.results_dir, "html")
    args.excel_dir = args.excel_dir or os.path.join(args.results_dir, "excel")
    args.image_dir = args.image_dir or os.path.join(args.results_dir, "images")
    main(args)
