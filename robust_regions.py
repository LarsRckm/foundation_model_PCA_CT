"""Robuste per-Winkel-Regionsgrenzen (Mandrel / Layer / Tab / Gehaeuse) als Ersatz
fuer die Generator-Funktion ``detect_regions_per_angle`` in der PCA-Pipeline.

Arbeitet spaltenweise auf dem BEREITS entrollten Polarbild (gleiche Aufruf-Signatur
wie die Generator-Funktion), sodass es in ``polar_embed.patch_regions`` ein Drop-in
ist. Verbesserungen gegenueber der reinen Gradienten-Peak-Suche:
  - Mandrel ueber zwei Cues (Intensitaetssprung + Oszillationsenergie), der
    zuverlaessigere gewinnt -> robust mit UND ohne Wickeltextur;
  - Gehaeuse = innere Kante der aeussersten hellen Wand; zusaetzlich wird ein davor
    liegender, durch einen Einbruch getrennter TAB erkannt (4. Kategorie);
  - beide Grenzkurven werden ueber den Winkel robust geglaettet (Hampel + zirkulaere
    Gauss-Glaettung).

Die Portierung stammt aus
``Data_Preprocessing/Maskierung/region_detection/robust_region_detection.py``.
Anders als dort wird hier NICHT neu um das Mandrel-Zentrum entrollt -- das Polarbild
kommt fertig aus der Embedding-Pipeline und muss mit den Patches ausgerichtet bleiben.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

DEFAULTS = dict(wall_rel=0.5, prom_rel=0.03, tab_near=0.88, tab_notch=0.05,
                hp_small=1.5, hp_large=8.0, rms_win=6, thr_frac=0.30, min_run=5,
                tab_conf=0.05)


# --------------------------------------------------------------------------- #
# per-Spalte Cues
# --------------------------------------------------------------------------- #
def _osc_energy(prof, hp_small=1.5, hp_large=8.0, rms_win=6):
    """Gleitende Hochpass-RMS (Wickel-Oszillation) entlang des Radius."""
    small = ndi.gaussian_filter1d(prof, hp_small)
    large = ndi.gaussian_filter1d(prof, hp_large)
    hp = small - large
    return np.sqrt(ndi.uniform_filter1d(hp * hp, rms_win) + 1e-12)


def _first_sustained(cond, hi, min_run, start=2):
    for i in range(start, hi - min_run):
        if cond[i] and np.all(cond[i:i + min_run]):
            return i
    return None


def _mandrel_boundary(prof_s, e, rv, r_ring, thr_frac=0.30, min_run=5, min_contrast=0.04):
    """Mandrel->Layer aus Intensitaetssprung ODER Oszillationsenergie (zuverlaessigerer)."""
    hi = int(max(10, min(r_ring, rv)))
    inner = prof_s[:max(5, int(0.12 * hi))]
    mand_lvl = float(np.median(inner))
    lay = prof_s[int(0.25 * hi):int(0.9 * hi)] if hi > 12 else prof_s[:hi]
    lay_lvl = float(np.median(lay)) if lay.size else mand_lvl
    contrast = lay_lvl - mand_lvl
    cand_i, conf_i = 0.0, 0.0
    if contrast > min_contrast:
        above = prof_s[:hi] > (mand_lvl + 0.5 * contrast)
        i = _first_sustained(above, hi, min_run)
        if i is not None:
            cand_i, conf_i = float(i), float(np.clip(contrast / (lay_lvl + 1e-6), 0, 1))
    lay_e = e[int(0.25 * hi):int(0.9 * hi)] if hi > 12 else e[:hi]
    ref = float(np.median(lay_e)) if lay_e.size else 0.0
    cand_e, conf_e = 0.0, 0.0
    if ref > 1e-6:
        i = _first_sustained(e > thr_frac * ref, hi, min_run)
        if i is not None:
            cand_e = float(i)
            conf_e = float(np.clip((ref - np.median(e[:max(1, i)])) / ref, 0, 1))
    return (cand_i, conf_i) if conf_i >= conf_e else (cand_e, conf_e)


def _outer_analysis(prof_s, rv, wall_rel=0.5, prom_rel=0.03, tab_near=0.88, tab_notch=0.05):
    """Gehaeusewand (aeusserstes helles Band) + optional ein davor liegender TAB
    (helles Band dicht an der Wand, durch einen Einbruch getrennt).

    Rueckgabe (r_housing, conf_housing, tab0, tab1, conf_tab); tab0/tab1 = -1 ohne Tab."""
    if rv < 12:
        return float(rv), 0.0, -1.0, -1.0, 0.0
    lo = int(0.4 * rv)
    seg = prof_s[lo:rv + 1]
    hi = float(prof_s[:rv + 1].max())
    lay = float(np.median(prof_s[int(0.2 * rv):int(0.8 * rv)]))
    span = hi - lay
    if span < 0.03 or seg.size < 6:
        return float(rv), 0.0, -1.0, -1.0, 0.0
    level_h = lay + wall_rel * span
    pk, _ = find_peaks(seg, height=level_h, prominence=prom_rel * span)
    if pk.size == 0:
        return float(rv), 0.0, -1.0, -1.0, 0.0
    pa = pk + lo
    wp = int(pa[np.argmax(pa)])

    def inner_edge(peak):
        i = peak
        while i - 1 >= 0 and prof_s[i - 1] >= level_h:
            i -= 1
        return i

    r_housing = float(inner_edge(wp))
    conf_h = float(np.clip(span, 0, 1))
    tab0 = tab1 = -1.0
    conf_t = 0.0
    innerk = [k for k in range(pa.size) if (pa[k] < wp - 2 and pa[k] >= tab_near * wp)]
    if innerk:
        tp = int(pa[innerk[int(np.argmax(prof_s[pa[innerk]]))]])
        between = prof_s[tp:wp + 1]
        vpos = tp + int(np.argmin(between))
        notch = float(min(prof_s[tp], prof_s[wp]) - between.min())
        if notch >= tab_notch * span:
            r_housing = float(vpos)
            a = tp
            while a - 1 >= 0 and prof_s[a - 1] >= level_h:
                a -= 1
            tab0, tab1 = float(a), float(max(a, vpos - 1))
            conf_t = float(np.clip(notch / (span + 1e-6) + 0.3, 0, 1)) * conf_h
    return r_housing, conf_h, tab0, tab1, conf_t


# --------------------------------------------------------------------------- #
# Winkel-Regularisierung (Hampel + zirkulaere Gauss-Glaettung)
# --------------------------------------------------------------------------- #
def _regularize(raw, conf, conf_min=0.08, med_frac=1.0 / 36, mad_k=3.5, sigma_frac=1.0 / 120):
    n = raw.size
    x = raw.astype(np.float64).copy()
    med_size = max(3, int(n * med_frac) | 1)
    med = ndi.median_filter(x, size=med_size, mode="wrap")
    resid = np.abs(x - med)
    mad = np.median(resid) + 1e-6
    outlier = (resid > mad_k * 1.4826 * mad) | (conf < conf_min)
    x[outlier] = med[outlier]
    sigma = max(1.0, n * sigma_frac)
    return ndi.gaussian_filter1d(x, sigma, mode="wrap").astype(np.float32)


def _clean_tab(t0, t1, c_t, r_m, r_r, conf_min=0.05, close_w=5):
    present = (c_t > conf_min) & (t0 >= 0)
    f = ndi.uniform_filter1d(present.astype(float), close_w, mode="wrap")
    present = f > 0.5
    tab0 = np.where(present, np.clip(t0, r_m + 1, r_r - 1), -1.0).astype(np.float32)
    tab1 = np.where(present, np.clip(t1, r_m + 1, r_r - 1), -1.0).astype(np.float32)
    return tab0, tab1


# --------------------------------------------------------------------------- #
# Drop-in fuer detect_regions_per_angle
# --------------------------------------------------------------------------- #
def detect_regions_per_angle(polar_img, padding_mask, r_valid_per_angle, r_scale=1.0,
                             return_tab=False, params=None):
    """Per-Winkel-Grenzen auf dem gegebenen Polarbild. Signatur kompatibel zur
    Generator-Funktion; ``return_tab=True`` liefert zusaetzlich das Tab-Band.

    Rueckgabe: (r_mandrel, r_ring)  oder  (r_mandrel, r_ring, tab0, tab1)
      -- je float32[N_theta], in nativen Polar-Zeilen (0..N_r); tab0/tab1 = -1 ohne Tab.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    N_r, N_theta = polar_img.shape
    r_valid_idx = np.clip(
        np.round(np.asarray(r_valid_per_angle, float) / max(1e-8, r_scale)).astype(int),
        1, N_r - 1)

    r_m = np.zeros(N_theta, np.float32); r_r = np.zeros(N_theta, np.float32)
    c_m = np.zeros(N_theta, np.float32); c_r = np.zeros(N_theta, np.float32)
    t0 = np.full(N_theta, -1.0, np.float32); t1 = np.full(N_theta, -1.0, np.float32)
    c_t = np.zeros(N_theta, np.float32)

    for j in range(N_theta):
        rv = int(r_valid_idx[j])
        prof = polar_img[:, j].astype(np.float64)
        prof_s = ndi.gaussian_filter1d(prof, 2.0)
        rh, crh, tb0, tb1, ctb = _outer_analysis(prof_s, rv, p["wall_rel"], p["prom_rel"],
                                                 p["tab_near"], p["tab_notch"])
        e = _osc_energy(prof, p["hp_small"], p["hp_large"], p["rms_win"])
        mi, cmi = _mandrel_boundary(prof_s, e, rv, rh, p["thr_frac"], p["min_run"])
        mi = min(mi, 0.85 * rh)
        r_m[j], r_r[j], c_m[j], c_r[j] = mi, rh, cmi, crh
        t0[j], t1[j], c_t[j] = tb0, tb1, ctb

    r_m = _regularize(r_m, c_m)
    r_r = _regularize(r_r, c_r)
    r_m = np.minimum(r_m, 0.9 * r_r).astype(np.float32)
    r_r = np.minimum(r_r, r_valid_idx.astype(np.float32)).astype(np.float32)
    tab0, tab1 = _clean_tab(t0, t1, c_t, r_m, r_r, p["tab_conf"])

    if return_tab:
        return r_m, r_r, tab0, tab1
    return r_m, r_r
