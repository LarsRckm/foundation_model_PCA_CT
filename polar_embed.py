"""
Shared helpers for the PCA-folder analyses: cartesian->polar transform (reusing
the generator's polar_transform.py read-only) + foundation-model patch-embedding
extraction. The polar framing and normalization mirror
embedding/create_embeddings.py EXACTLY, so the embeddings produced here match the
large-scale pipeline for the same model.

Used by polar_pca_eval.py; nothing in the generator is modified.
"""

import os
import re
import sys

import numpy as np
import torch
from PIL import Image

# Same registry as embedding/create_embeddings.py: name -> (hf_id, patch_size).
MODELS = {
    "dinov2-small": ("facebook/dinov2-small", 14),
    "dinov2-base":  ("facebook/dinov2-base",  14),
    "dinov2-large": ("facebook/dinov2-large", 14),
    "dinov2-giant": ("facebook/dinov2-giant", 14),
    "dinov3-large": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16),
    "rad-dino":     ("microsoft/rad-dino",    14),
}

DEFAULT_POLAR = r"C:\Users\larsr\Documents\Uni\Masterarbeit\Masterarbeit_cartesianApproach\CT_scan_model"
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST


# --------------------------------------------------------------------------- #
# Polar transform (identical to create_embeddings.py)
# --------------------------------------------------------------------------- #
def load_polar_fns(polar_dir):
    """Import detect_cell_boundary / cart_to_polar_boundary from the generator
    (read-only). Needed only for cartesian inputs."""
    if polar_dir and polar_dir not in sys.path:
        sys.path.insert(0, polar_dir)
    from polar_transform import detect_cell_boundary, cart_to_polar_boundary
    return detect_cell_boundary, cart_to_polar_boundary


def load_region_fn(polar_dir=None):
    """Load the ROBUST per-angle region detector (local ``robust_regions.py``), which
    REPLACES the generator's ``detect_regions_per_angle``.

    It returns four boundary curves (r_mandrel, r_ring, tab0, tab1) and distinguishes
    mandrel / jelly-roll layers / tab / cell housing (+ padding). ``polar_dir`` is kept
    for backward compatibility but is no longer used."""
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from robust_regions import detect_regions_per_angle
    return detect_regions_per_angle


def frame_size(n_r, n_theta, patch):
    """Nearest multiple of patch to (n_r, n_theta) -> embedding grid fits exactly."""
    return max(patch, round(n_r / patch) * patch), max(patch, round(n_theta / patch) * patch)


def _cart_to_polar(gray_u8, n_r, n_theta, detect_fn, cart_fn):
    cx, cy, r_valid = detect_fn(gray_u8, N_theta=n_theta)[:3]
    r_valid = np.asarray(r_valid, np.float32)
    img = gray_u8.astype(np.float32) / 127.5 - 1.0
    polar, pad_mask, _r = cart_fn(img, cx, cy, r_valid, n_r, n_theta)
    disp = np.clip((polar + 1.0) / 2.0, 0.0, 1.0)
    disp[polar < -1.5] = 0.0
    return (disp * 255).astype(np.uint8), pad_mask.astype(bool)


def _reframe(arr_u8, H, W, nearest=False):
    im = Image.fromarray(arr_u8)
    return np.asarray(im.resize((W, H), NEAREST if nearest else BICUBIC))


def _patch_mask(valid_HW, patch):
    H, W = valid_HW.shape
    gh, gw = H // patch, W // patch
    v = valid_HW[: gh * patch, : gw * patch].reshape(gh, patch, gw, patch)
    return (v.mean(axis=(1, 3)) > 0.5).reshape(-1)


def to_polar_display(path, fmt, n_r, n_theta, H, W, patch, fns, pad_thresh=5):
    """Load one image and return (disp_rgb uint8 [H,W,3], pmask [gh*gw] bool).

    disp is the polar image reframed to (H, W); pmask marks patches inside the
    cell (vs. padding). Pipeline identical to create_embeddings.py's dataset."""
    gray = np.asarray(Image.open(path).convert("L"), np.uint8)
    if fmt == "cart":
        if fns is None:
            raise RuntimeError("cartesian input needs polar_transform fns (load_polar_fns).")
        disp, valid = _cart_to_polar(gray, n_r, n_theta, *fns)
    else:                                   # already polar
        disp = np.asarray(Image.fromarray(gray).resize((n_theta, n_r), BICUBIC))
        valid = disp > pad_thresh
        if valid.mean() > 0.995:
            valid = np.ones_like(disp, bool)
    disp_r = _reframe(disp, H, W, nearest=False)
    valid_r = _reframe((valid.astype(np.uint8) * 255), H, W, nearest=True) > 127
    pmask = _patch_mask(valid_r, patch)
    rgb = np.repeat(disp_r[:, :, None], 3, axis=2).astype(np.uint8)
    return rgb, pmask


# --------------------------------------------------------------------------- #
# Model loading + patch-embedding extraction
# --------------------------------------------------------------------------- #
def load_model(model_name, device=None, token=None):
    """Return (model, processor, patch_size, device) for a registry name.

    ``token`` is an optional Hugging Face access token for GATED models
    (e.g. ``dinov3-large``). When it is None, transformers still falls back to the
    ``HF_TOKEN`` environment variable or a cached ``huggingface-cli login``, so the
    token only needs to be passed explicitly (via ``--hf_token``) when neither of
    those is set up."""
    from transformers import AutoImageProcessor, AutoModel
    hf_id, patch = MODELS[model_name]
    auth = {"token": token} if token else {}
    try:
        proc = AutoImageProcessor.from_pretrained(hf_id, **auth)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(hf_id, **auth).to(device).eval()
    except Exception as exc:                       # gated repo / missing access / old transformers
        raise RuntimeError(
            f"Could not load model '{model_name}' ({hf_id}): {exc}\n"
            "If this is a gated model (e.g. dinov3-large): request access to the repo "
            "on huggingface.co, then provide a token via --hf_token, the HF_TOKEN "
            "environment variable, or `huggingface-cli login`. Also make sure your "
            "installed `transformers` is recent enough to support the model.") from exc
    return model, proc, patch, device


def embed_image(path, fmt, model, proc, patch, device, n_r, n_theta, fns, pad_thresh=5):
    """One image -> (patches [N, D] float32, grid_h, grid_w, pmask [N] bool, disp_pil).

    N = grid_h * grid_w patch tokens of the (reframed) polar image. Normalization
    and token selection (the LAST n_tok tokens, robust to CLS/register tokens)
    match create_embeddings.py."""
    H, W = frame_size(n_r, n_theta, patch)
    gh, gw = H // patch, W // patch
    n_tok = gh * gw

    disp_rgb, pmask = to_polar_display(path, fmt, n_r, n_theta, H, W, patch, fns, pad_thresh)
    rgb = disp_rgb.astype(np.float32) / 255.0
    mean = np.asarray(proc.image_mean, np.float32)
    std = np.asarray(proc.image_std, np.float32)
    x = (rgb - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            out = model(pixel_values=x, interpolate_pos_encoding=True)
        except TypeError:
            out = model(pixel_values=x)
    patches = out.last_hidden_state[0, -n_tok:, :].float().cpu().numpy()
    return patches, gh, gw, pmask, Image.fromarray(disp_rgb)


# --------------------------------------------------------------------------- #
# Region classification (mandrel / layers / housing) for the patch grid
# --------------------------------------------------------------------------- #
# Region codes (0..3 as in dataset_ct_polar._compute_region_labels; 4 = tab, viz-only).
REGION_PADDING, REGION_MANDREL, REGION_LAYERS, REGION_HOUSING, REGION_TAB = 0, 1, 2, 3, 4


def _native_for_regions(path, fmt, n_r, n_theta, fns, pad_thresh=5):
    """Rebuild the native polar field + geometry that detect_regions_per_angle
    needs, matching to_polar_display / _cart_to_polar EXACTLY so the regions line
    up with the embeddings.

    Returns (polar_float [n_r, n_theta], valid [n_r, n_theta] bool,
             r_valid_per_angle [n_theta] float32, r_scale float)."""
    gray = np.asarray(Image.open(path).convert("L"), np.uint8)
    if fmt == "cart":
        if fns is None:
            raise RuntimeError("cartesian input needs polar_transform fns (load_polar_fns).")
        detect_fn, cart_fn = fns
        cx, cy, r_valid = detect_fn(gray, N_theta=n_theta)[:3]
        r_valid = np.asarray(r_valid, np.float32)
        img = gray.astype(np.float32) / 127.5 - 1.0
        polar, pad_mask, r_max = cart_fn(img, cx, cy, r_valid, n_r, n_theta)
        valid = pad_mask.astype(bool)
        r_scale = float(r_max) / max(1.0, float(n_r - 1))
    else:                                   # already polar
        disp = np.asarray(Image.fromarray(gray).resize((n_theta, n_r), BICUBIC))
        valid = disp > pad_thresh
        if valid.mean() > 0.995:
            valid = np.ones_like(disp, bool)
        polar = disp.astype(np.float32) / 127.5 - 1.0
        # No physical radial scale for native-polar input: use r_scale = 1 and give
        # the outermost valid row per column as r_valid, so detect_regions_per_angle
        # reads each column up to its real cell edge (r_valid_idx == that row).
        rows = np.arange(n_r, dtype=np.float32)[:, None]
        rr = np.where(valid, rows, -1.0).max(axis=0)
        rr[rr < 0] = float(n_r - 1)
        r_valid = rr.astype(np.float32)
        r_scale = 1.0
    return polar.astype(np.float32), valid, r_valid.astype(np.float32), r_scale


def patch_regions(path, fmt, n_r, n_theta, H, W, patch, fns, region_fn, pad_thresh=5):
    """Classify every patch token of one image into mandrel / layers / housing and
    return the two transition curves for drawing on the displayed polar image.

    Uses the robust local ``detect_regions_per_angle`` (``robust_regions``, passed as
    ``region_fn``), which also detects the tab.

    Returns
    -------
    regions : int8 [gh*gw], row-major like the patch tokens
              (0=padding, 1=mandrel, 2=layers, 3=housing, 4=tab)
    bounds  : dict {"m": [[x,y],...], "r": [[x,y],...]} — the mandrel/layer and
              layer/housing transition polylines in DISPLAY-image fractions
              (x = angle 0..1, y = radius 0..1), ready to overlay on the polar image.
    """
    polar, valid, r_valid, r_scale = _native_for_regions(path, fmt, n_r, n_theta, fns, pad_thresh)
    pad_mask_f = valid.astype(np.float32)

    # Robust per-angle detector (robust_regions): four curves incl. the tab band.
    r_mandrel, r_ring, tab0, tab1 = region_fn(polar, pad_mask_f, r_valid,
                                              r_scale=r_scale, return_tab=True)
    r_mandrel = np.minimum(np.asarray(r_mandrel, np.float32),
                           np.asarray(r_ring, np.float32))   # safety: mandrel <= ring
    tab0 = np.asarray(tab0, np.float32)
    tab1 = np.asarray(tab1, np.float32)

    gh, gw = H // patch, W // patch

    # Patch centres, mapped from the H x W display frame back to native polar units.
    row_c = (np.arange(gh, dtype=np.float32) + 0.5) * patch * (n_r / float(H))        # [gh]
    col_c = (np.arange(gw, dtype=np.float32) + 0.5) * patch * (n_theta / float(W))    # [gw]
    jcol = np.clip(np.round(col_c).astype(int), 0, n_theta - 1)                        # [gw]
    m_at = r_mandrel[jcol]                                                             # [gw]
    r_at = np.asarray(r_ring, np.float32)[jcol]                                        # [gw]
    t0_at = tab0[jcol]                                                                 # [gw]
    t1_at = tab1[jcol]                                                                 # [gw]

    # Partition by radius (rows < mandrel / < ring / else); tab overrides the layer
    # patches inside its band where present (tab0 >= 0).
    rc = row_c[:, None]                                                                # [gh,1]
    is_mandrel = rc < m_at[None, :]
    is_housing = rc >= r_at[None, :]
    lab = np.where(is_mandrel, REGION_MANDREL,
                   np.where(is_housing, REGION_HOUSING, REGION_LAYERS)).astype(np.int8)
    is_tab = ((t0_at[None, :] >= 0) & (rc >= t0_at[None, :]) & (rc <= t1_at[None, :])
              & (lab == REGION_LAYERS))
    lab[is_tab] = REGION_TAB
    regions = lab.reshape(-1).copy()                                                   # row-major

    # Padding patches: identical to embed_image's pmask (reframed valid mask).
    valid_r = _reframe((valid.astype(np.uint8) * 255), H, W, nearest=True) > 127
    pmask = _patch_mask(valid_r, patch)
    regions[~pmask] = REGION_PADDING

    # Transition polylines (subsample the angle axis to keep the HTML small).
    step = max(1, n_theta // 360)
    js = np.arange(0, n_theta, step)
    xs = js / float(n_theta)
    ym = np.clip(np.asarray(r_mandrel, np.float32)[js] / float(n_r), 0.0, 1.0)
    yr = np.clip(np.asarray(r_ring, np.float32)[js] / float(n_r), 0.0, 1.0)
    bounds = {
        "m": [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, ym)],
        "r": [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, yr)],
    }
    return regions, bounds
