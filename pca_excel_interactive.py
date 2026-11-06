"""
PCA of DINOv2 patch embeddings -> (1) Excel report of components + variance,
                                  (2) interactive Plotly 3D scatter with hover-image.

Reads two images exactly like cluster_ct_scans.py (facebook/dinov2-small, patch 14).
The Excel report fits a separate PCA per image. The 3D plot uses a PCA basis fit on
the REAL patch tokens only and projects the synthetic tokens onto it (see --html_out).

Outputs:
  --excel_out  (default pca_variance.xlsx)
      One row per principal component with explained variance (%), cumulative (%),
      and the eigenvalue (variance). This is the direct answer to "which PC keeps
      how much percent of the variance".

  --html_out   (default pca_interactive_3d.html)
      Self-contained interactive 3D scatter (PC1/PC2/PC3), points colored by
      real vs. synthetic. Hovering a point shows, in a side panel, the *processed*
      input image with the corresponding patch marked by a red box.
      The 3 axes are the PCA basis fit on the REAL embeddings only; the synthetic
      patches are projected onto them (scalar product after subtracting the real
      mean), like PCA_comparison. Marker size is constant; a "Patches (horizontal)"
      control thins the cloud and the "Regionen" checkbox colors points by
      mandrel/layer/housing.
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

COLOR_REAL = "#2c7fb8"
COLOR_SYNTH = "#d95f0e"


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


def extract(model, processor, image, patch_size=14):
    """Return (patch_embeddings [N, D], grid_h, grid_w, processed_pil_image)."""
    inputs = processor(images=image, return_tensors="pt")
    h, w = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
    grid_h, grid_w = h // patch_size, w // patch_size
    with torch.no_grad():
        outputs = model(**inputs)
    patches = outputs.last_hidden_state[0, 1:, :].numpy()
    proc_img = processed_image_to_pil(inputs["pixel_values"], processor)
    if grid_h * grid_w != patches.shape[0]:
        print(f"Warning: grid {grid_h}x{grid_w} != token count {patches.shape[0]}")
    return patches, grid_h, grid_w, proc_img


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Excel report
# --------------------------------------------------------------------------- #
def _variance_rows(pca):
    evr = pca.explained_variance_ratio_
    eig = pca.explained_variance_
    cum = np.cumsum(evr)
    n = len(evr)
    rows = {
        "Principal Component": [f"PC{i+1}" for i in range(n)],
        "Explained Variance (%)": np.round(evr * 100, 4),
        "Cumulative Variance (%)": np.round(cum * 100, 4),
        "Eigenvalue (variance)": np.round(eig, 6),
    }
    return rows, cum, n


def _sanitize_sheet_name(name):
    for ch in r"[]:*?/\\":
        name = name.replace(ch, "_")
    return name[:31]


def write_excel(sheets, path):
    """sheets: list of (sheet_name, fitted PCA). One sheet per image."""
    try:
        import pandas as pd
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, pca in sheets:
                rows, _, _ = _variance_rows(pca)
                pd.DataFrame(rows).to_excel(
                    writer, index=False, sheet_name=_sanitize_sheet_name(name))
    except ImportError:
        # Fallback to openpyxl directly if pandas is unavailable.
        from openpyxl import Workbook
        wb = Workbook()
        for i, (name, pca) in enumerate(sheets):
            rows, _, _ = _variance_rows(pca)
            ws = wb.active if i == 0 else wb.create_sheet()
            ws.title = _sanitize_sheet_name(name)
            headers = list(rows.keys())
            ws.append(headers)
            for r in range(len(rows[headers[0]])):
                ws.append([rows[h][r] for h in headers])
        wb.save(path)

    print(f"  saved {path}")
    for name, pca in sheets:
        _, cum, n = _variance_rows(pca)
        print(f"  [{name}] {n} components; PC1-3 keep {cum[2]*100:.1f}%; "
              f"{int(np.searchsorted(cum, 0.90) + 1)} PCs reach 90%.")


# --------------------------------------------------------------------------- #
# Interactive 3D Plotly HTML (built manually for the hover-image side panel)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DINOv2 PCA - interactive 3D</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 12px; }
  #wrap { display: flex; gap: 16px; align-items: flex-start; }
  #plot { flex: 1; height: 720px; }
  #side { width: 360px; flex: none; }
  #imgWrap { position: relative; width: 340px; border: 1px solid #ccc; }
  #imgWrap img { width: 100%; display: block; }
  #patchBox { position: absolute; border: 2px solid red; box-shadow: 0 0 0 1px white;
              display: none; pointer-events: none; }
  #bounds { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
            display: none; pointer-events: none; }
  .sliderRow { margin: 10px 0; }
  .sliderRow label { display: inline-block; width: 130px; font-size: 13px; }
  #regionRow label { width: auto; cursor: pointer; }
  input[type=range] { width: 180px; vertical-align: middle; }
  #title { font-weight: 600; margin-bottom: 6px; }
  .hint { color: #666; font-size: 12px; }
  .legend { font-size: 12px; margin-top: 6px; display: none; }
  .legend span { margin-right: 10px; white-space: nowrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; margin-right: 3px;
              vertical-align: middle; border: 1px solid #999; }
  .legend u { display: inline-block; width: 14px; height: 0; margin-right: 3px;
              vertical-align: middle; }
</style>
</head>
<body>
<h2>DINOv2 patch embeddings - PCA (2D PC1/PC2 oder 3D PC1/PC2/PC3)</h2>
<p class="hint">Axes = PCA basis of the REAL embeddings (PC1/PC2/PC3); synthetic points are
   projected onto it &middot; color = real vs. synthetic &middot;
   "Patches (horizontal)" thins points along the angle axis (every 2nd/3rd/4th column)
   &middot; the "Regionen" checkbox colors points by
   mandrel/layer/housing (dark tone = real, light = synthetic) and draws the
   transitions on the polar image &middot; hover a point to mark its patch.</p>
<div id="wrap">
  <div id="plot"></div>
  <div id="side">
    <div class="sliderRow" id="viewRow">
      <label>Ansicht:</label>
      <label style="width:auto;cursor:pointer"><input type="radio" name="viewMode" value="3d" checked> 3D (PC1/PC2/PC3)</label>
      <label style="width:auto;cursor:pointer"><input type="radio" name="viewMode" value="2d"> 2D (PC1/PC2)</label>
    </div>
    <div id="title">Hover a point...</div>
    <div id="imgWrap">
      <img id="hoverImg"/>
      <div id="patchBox"></div>
      <svg id="bounds" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline id="bRing" fill="none" stroke="#00e5ff" stroke-width="2"
                  vector-effect="non-scaling-stroke"/>
        <polyline id="bMand" fill="none" stroke="#ffeb3b" stroke-width="2"
                  vector-effect="non-scaling-stroke"/>
      </svg>
    </div>
    <div class="sliderRow" id="regionRow">
      <label><input type="checkbox" id="showRegions">
        Regionen einf&auml;rben &amp; &Uuml;berg&auml;nge einzeichnen</label>
      <div class="legend" id="regionLegend">
        <div style="margin-bottom:3px;color:#444">dunkler Ton = echt &middot; heller Ton = synthetisch</div>
        <span><i style="background:#1b7a37;margin-right:1px"></i><i style="background:#8ede93"></i>Mandrel</span>
        <span><i style="background:#5e3370;margin-right:1px"></i><i style="background:#c7a6d8"></i>Layer</span>
        <span><i style="background:#a11117;margin-right:1px"></i><i style="background:#f2898e"></i>Zellgeh&auml;use</span>
        <span><i style="background:#b5680f;margin-right:1px"></i><i style="background:#f0b35a"></i>Tab</span>
        <span><i style="background:#777777;margin-right:1px"></i><i style="background:#cfcfcf"></i>Padding</span>
        <br>
        <span><u style="border-top:3px solid #ffeb3b"></u>Mandrel&rarr;Layer</span>
        <span><u style="border-top:3px solid #00e5ff"></u>Layer&rarr;Geh&auml;use</span>
      </div>
    </div>
    <div class="sliderRow">
      <label>Patches (horizontal):</label>
      <select id="colStride">
        <option value="1">alle</option>
        <option value="2">jeder 2.</option>
        <option value="3">jeder 3.</option>
        <option value="4">jeder 4.</option>
      </select>
    </div>
    <div class="hint" id="count"></div>
  </div>
</div>

<script>
var traces  = __TRACES__;
var layout3d = __LAYOUT__;
var layout2d = __LAYOUT2D__;
var images  = __IMAGES__;
var regionFlag = __REGION_FLAG__; // true when mandrel/layer/housing data is available
var boundData  = __BOUNDS__;      // {0:{m:[[x,y]..],r:[..]}, 1:{...}} or null
var names = traces.map(function(t){ return t.name; });
var HT3 = "PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>PC3 %{z:.2f}<br>patch r%{customdata[5]} c%{customdata[6]}<extra></extra>";
var HT2 = "PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>patch r%{customdata[5]} c%{customdata[6]}<extra></extra>";
var mode = '3d';   // '3d' = PC1/PC2/PC3 scatter3d, '2d' = PC1/PC2 scatter

// region code (customdata[7]) -> [echt (dunkler Ton), synthetisch (hellerer Ton)].
// Same hue per region; the dark/light tone keeps real vs. synthetic distinguishable.
var REGION_COLORS = {
  0: ['#777777', '#cfcfcf'],   // Padding   (grau)
  1: ['#1b7a37', '#8ede93'],   // Mandrel   (gruen)
  2: ['#5e3370', '#c7a6d8'],   // Layer     (lila)
  3: ['#a11117', '#f2898e'],   // Gehaeuse  (rot)
  4: ['#b5680f', '#f0b35a']    // Tab       (orange)
};
var REGION_NAMES  = {0:'Padding', 1:'Mandrel', 2:'Layer', 3:'Zellgehaeuse', 4:'Tab'};

// Keep an untouched copy of every trace so filtering is reversible.
var full = traces.map(function(t){
  return { x:t.x.slice(), y:t.y.slice(), z:t.z.slice(),
           size:t.marker.size.slice(),
           cd:t.customdata.map(function(a){ return a.slice(); }),
           baseColor:t.marker.color };
});

function regionsActive(){
  var cb = document.getElementById('showRegions');
  return regionFlag && cb && cb.checked;
}

// ---- draw the mandrel->layer and layer->housing transitions on the polar image ----
function ptsToStr(pts){
  return pts.map(function(p){ return (p[0]*100).toFixed(2)+','+(p[1]*100).toFixed(2); }).join(' ');
}
function drawBounds(){
  var svg = document.getElementById('bounds');
  if (!svg) return;
  var on = regionsActive() && boundData;
  svg.style.display = on ? 'block' : 'none';
  if (!on) return;
  var b = boundData[curImg] || {m:[], r:[]};
  document.getElementById('bMand').setAttribute('points', ptsToStr(b.m));
  document.getElementById('bRing').setAttribute('points', ptsToStr(b.r));
}

// ---- build traces for the current view; honours the stride + "Regionen" recoloring ----
// 3D -> scatter3d (PC1/PC2/PC3); 2D -> scattergl (PC1/PC2 only).
function buildTraces(){
  var stride = parseInt(document.getElementById('colStride').value, 10) || 1;
  var byRegion = regionsActive();
  var out = [], shown = 0;
  for (var i = 0; i < full.length; i++){
    var f = full[i], xs=[], ys=[], zs=[], ss=[], cds=[], cols=[];
    for (var k = 0; k < f.x.length; k++){
      // col = patch column = angle axis (horizontal); stride thins it (every Nth column).
      if (f.cd[k][6] % stride === 0){
        xs.push(f.x[k]); ys.push(f.y[k]); zs.push(f.z[k]);
        ss.push(f.size[k]); cds.push(f.cd[k]);
        // region hue + dark(echt)/light(synth) tone via cd[0] (0=real, 1=synth).
        if (byRegion) cols.push(REGION_COLORS[f.cd[k][7]][f.cd[k][0]]);
      }
    }
    shown += xs.length;
    var color = byRegion ? cols : f.baseColor;
    if (mode === '3d'){
      out.push({ type:'scatter3d', mode:'markers', name:names[i],
                 x:xs, y:ys, z:zs, customdata:cds,
                 marker:{ size:ss, color:color, opacity:0.75 }, hovertemplate:HT3 });
    } else {
      out.push({ type:'scattergl', mode:'markers', name:names[i],
                 x:xs, y:ys, customdata:cds,
                 marker:{ size:ss.map(function(s){ return s*1.7; }), color:color, opacity:0.75 },
                 hovertemplate:HT2 });
    }
  }
  document.getElementById('count').innerText = shown + ' points shown (stride ' + stride + ')';
  return out;
}

function render(){
  Plotly.react('plot', buildTraces(), mode === '3d' ? layout3d : layout2d, {responsive:true});
  drawBounds();
}

var gd = document.getElementById('plot');
var curImg = 0;   // image currently shown in the side panel
Plotly.newPlot('plot', buildTraces(), layout3d, {responsive:true});

// ---- hover: show processed image with the patch marked ----
gd.on('plotly_hover', function(ev){
  var cd = ev.points[0].customdata;  // [imgId,x0,y0,x1,y1,row,col,region]
  var imgId = cd[0];
  curImg = imgId;
  document.getElementById('hoverImg').src = images[imgId];
  var regTxt = regionFlag ? '  [' + REGION_NAMES[cd[7]] + ']' : '';
  document.getElementById('title').innerText =
      (imgId === 0 ? 'Real scan' : 'Synthetic scan') +
      '  (patch row ' + cd[5] + ', col ' + cd[6] + ')' + regTxt;
  var box = document.getElementById('patchBox');
  box.style.left   = (cd[1]*100) + '%';
  box.style.top    = (cd[2]*100) + '%';
  box.style.width  = ((cd[3]-cd[1])*100) + '%';
  box.style.height = ((cd[4]-cd[2])*100) + '%';
  box.style.display = 'block';
  drawBounds();
});

function applyRegionMode(){
  var leg = document.getElementById('regionLegend');
  if (leg) leg.style.display = regionsActive() ? 'block' : 'none';
  render();   // recolor + redraw transitions (keeps the current 2D/3D view)
}

document.getElementById('colStride').addEventListener('change', render);

// 2D / 3D view toggle
var vradios = document.querySelectorAll('input[name="viewMode"]');
for (var vi = 0; vi < vradios.length; vi++){
  vradios[vi].addEventListener('change', function(e){ mode = e.target.value; render(); });
}

if (regionFlag && document.getElementById('showRegions')){
  document.getElementById('showRegions').addEventListener('change', applyRegionMode);
} else {
  var rr = document.getElementById('regionRow');
  if (rr) rr.style.display = 'none';   // no region data (e.g. dinov2-small CLI)
}

// initial state
document.getElementById('hoverImg').src = images[0];
drawBounds();
</script>
</body>
</html>
"""


def build_interactive_html(scores, labels, grids, img_b64, evr, path,
                           regions=None, bounds=None):
    """scores: (N, >=3), only PC1..PC3 used; labels: 0/1; grids: (grid_h, grid_w) per image.

    regions : optional [regions_real, regions_synth], each an int array (row-major,
              0=padding/1=mandrel/2=layers/3=housing) aligned with that image's patch
              tokens. When given, a checkbox colors the points by region and draws the
              mandrel/layer and layer/housing transitions on the polar image.
    bounds  : optional [bounds_real, bounds_synth], each {"m":[[x,y]..], "r":[[x,y]..]}
              transition polylines in display-image fractions."""
    # Strictly 3 components -> no PC4; marker size is constant.
    sizes_all = np.full(scores.shape[0], 4.5)

    traces = []
    for grp, name, color in [(0, "Real", COLOR_REAL), (1, "Synthetic", COLOR_SYNTH)]:
        idx = np.where(labels == grp)[0]
        gh, gw = grids[grp]
        reg = regions[grp] if regions is not None else None
        cd = []
        for local_p, gi in enumerate(idx):
            r = local_p // gw
            c = local_p % gw
            x0, x1 = c / gw, (c + 1) / gw
            y0, y1 = r / gh, (r + 1) / gh
            region_val = int(reg[local_p]) if reg is not None else 0
            # customdata = [grp, x0, y0, x1, y1, row, col, region]
            cd.append([grp, x0, y0, x1, y1, int(r), int(c), region_val])
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "name": name,
            "x": scores[idx, 0].tolist(),
            "y": scores[idx, 1].tolist(),
            "z": scores[idx, 2].tolist(),
            "customdata": cd,
            "marker": {"size": sizes_all[idx].tolist(), "color": color, "opacity": 0.75},
            "hovertemplate": ("PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>PC3 %{z:.2f}"
                              "<br>patch r%{customdata[5]} c%{customdata[6]}<extra></extra>"),
        })

    layout = {
        "scene": {
            "xaxis": {"title": f"PC1 ({evr[0]*100:.1f}%)"},
            "yaxis": {"title": f"PC2 ({evr[1]*100:.1f}%)"},
            "zaxis": {"title": f"PC3 ({evr[2]*100:.1f}%)"},
        },
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "legend": {"x": 0, "y": 1},
    }
    # 2D view: only the first two PCs of the real basis.
    layout2d = {
        "xaxis": {"title": f"PC1 ({evr[0]*100:.1f}%)", "zeroline": False},
        "yaxis": {"title": f"PC2 ({evr[1]*100:.1f}%)", "zeroline": False},
        "margin": {"l": 55, "r": 10, "t": 10, "b": 45},
        "legend": {"x": 0, "y": 1},
        "hovermode": "closest",
    }

    images = {0: img_b64[0], 1: img_b64[1]}

    region_flag = regions is not None
    bound_data = None
    if bounds is not None:
        bound_data = {0: bounds[0], 1: bounds[1]}

    html = (HTML_TEMPLATE
            .replace("__TRACES__", json.dumps(traces))
            .replace("__LAYOUT2D__", json.dumps(layout2d))
            .replace("__LAYOUT__", json.dumps(layout))
            .replace("__IMAGES__", json.dumps(images))
            .replace("__REGION_FLAG__", json.dumps(region_flag))
            .replace("__BOUNDS__", json.dumps(bound_data)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  saved {path}")


def save_2d_png(scores, labels, evr, path):
    """Static PNG of the 2D view: PC1 vs PC2 of the REAL basis, colored real vs. synthetic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, name, color in [(0, "Real", COLOR_REAL), (1, "Synthetic", COLOR_SYNTH)]:
        m = labels == grp
        ax.scatter(scores[m, 0], scores[m, 1], s=6, alpha=0.6, c=color,
                   label=name, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("DINOv2 PCA (PC1/PC2) - real basis")
    ax.legend(markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
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

    print("Loading images...")
    img_real = load_image(args.real_image)
    img_synth = load_image(args.synthetic_image)

    print("Extracting DINOv2 patch embeddings...")
    patches_real, gh_r, gw_r, proc_real = extract(model, processor, img_real)
    patches_synth, gh_s, gw_s, proc_synth = extract(model, processor, img_synth)
    run_excel(patches_real, gh_r, gw_r, proc_real,
              patches_synth, gh_s, gw_s, proc_synth,
              os.path.basename(args.real_image), os.path.basename(args.synthetic_image), args)


def run_excel(patches_real, gh_r, gw_r, proc_real,
              patches_synth, gh_s, gw_s, proc_synth, real_name, synth_name, args,
              regions=None, bounds=None):
    """Per-image PCA (Excel) + real-basis PCA projection (interactive 3D HTML).
    Model-agnostic: used by the CLI below (dinov2-small) and by polar_pca_eval.py.

    The 3D plot does NOT fit a joint real+synth PCA. The basis is fit on the REAL
    embeddings only; the synthetic patches are projected onto the first 3 real axes
    (scalar product after subtracting the real mean), like PCA_comparison -> real sits
    at the origin, synthetic shares the same axes.

    regions/bounds (optional): per-image mandrel/layer/housing labels and transition
    polylines; when supplied, the HTML gains a checkbox to color points by region and
    draw the transitions on the polar image. The dinov2-small CLI leaves them None."""
    X = np.vstack((patches_real, patches_synth))
    labels = np.concatenate([np.zeros(patches_real.shape[0]), np.ones(patches_synth.shape[0])])
    print(f"Feature matrix: {X.shape[0]} patches x {X.shape[1]} dims "
          f"(real={patches_real.shape[0]}, synth={patches_synth.shape[0]})")

    # Per-image PCA -> one Excel sheet each (each image analyzed on its own).
    print("Fitting per-image PCA...")
    pca_real = PCA(svd_solver="full", random_state=42).fit(patches_real)
    pca_synth = PCA(svd_solver="full", random_state=42).fit(patches_synth)

    print("\nWriting Excel report...")
    write_excel([(real_name, pca_real), (synth_name, pca_synth)], args.excel_out)

    # 3D plot: REAL PCA is the basis; project BOTH groups onto its first 3 axes
    # (= (X - real_mean) @ real_components[:3].T). Reuses pca_real so the plot axes
    # match the "real" Excel sheet exactly.
    print("\nProjecting both groups onto the REAL PCA basis (3 components)...")
    if pca_real.components_.shape[0] < 3:
        print("Real embeddings yield < 3 components; skipping HTML.")
        return
    scores = pca_real.transform(X)[:, :3]              # (N, 3)
    evr = pca_real.explained_variance_ratio_[:3]       # explained variance OF THE REAL data

    print("\nBuilding interactive HTML (2D/3D toggle)...")
    img_b64 = [pil_to_base64(proc_real), pil_to_base64(proc_synth)]
    build_interactive_html(scores, labels, [(gh_r, gw_r), (gh_s, gw_s)], img_b64,
                           evr, args.html_out, regions=regions, bounds=bounds)

    # Static PNG of the 2D view (PC1/PC2). Default next to the HTML if --png_out is absent
    # (polar_pca_eval.py does not define --png_out).
    png_out = getattr(args, "png_out", None) or (os.path.splitext(args.html_out)[0] + "_2d.png")
    print("\nSaving 2D PNG...")
    save_2d_png(scores, labels, evr, png_out)

    print(f"\nDone. Open '{args.html_out}' in a browser (toggle 2D/3D); 2D PNG at '{png_out}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA of DINOv2 patch embeddings: Excel variance report + interactive 3D plot.")
    parser.add_argument("--real_image", type=str, default="PCA/real.png",
                        help="Path to the real CT scan image (PNG)")
    parser.add_argument("--synthetic_image", type=str, default="PCA/fake.png",
                        help="Path to the synthetic CT scan image (PNG)")
    parser.add_argument("--excel_out", type=str, default="PCA/pca_variance.xlsx",
                        help="Output path for the Excel variance report")
    parser.add_argument("--html_out", type=str, default="PCA/pca_interactive_3d.html",
                        help="Output path for the interactive Plotly HTML (2D/3D toggle)")
    parser.add_argument("--png_out", type=str, default=None,
                        help="Output path for the static 2D PNG (default: <html_out>_2d.png)")
    args = parser.parse_args()
    main(args)
