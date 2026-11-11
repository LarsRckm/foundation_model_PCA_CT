# PCA von Foundation-Model-Patch-Embeddings (CT-Scans)

PCA-Analyse und Visualisierung von Patch-Embeddings aus Foundation-Modellen
(DINOv2 / DINOv3, `rad-dino`) für echte vs. synthetische CT-Scans. Teil der
Masterarbeit *Foundation Model Clustering*.

Die Skripte extrahieren Patch-Token eines Bildpaares (real + synthetisch),
projizieren sie per PCA und beantworten damit konkrete Fragen an die
Dimensionsreduktion – inklusive einer interaktiven 3D-Darstellung.

## Skripte

| Datei | Zweck |
|-------|-------|
| `polar_pca_eval.py` | Haupteinstiegspunkt: real + synthetisches Bild → (optional) Kartesisch→Polar-Transform → Embeddings aus **einem oder mehreren** Foundation-Modellen → Violin- und/oder Excel+3D-Analyse (ein HTML je Modell, eine Excel mit allen Modellen, ein Modell-Vergleichsbild). |
| `polar_embed.py` | Gemeinsame Helfer: Polar-Transform + Patch-Embedding-Extraktion; Modell-Registry (DINOv2 small/base/large/giant, DINOv3-large, rad-dino). |
| `pca_cumulative_violin.py` | Split-Violin-Plot der kumulativen erklärten Varianz (real vs. synthetisch). |
| `pca_excel_interactive.py` | Excel-Report (Varianz je Hauptkomponente) + interaktiver Plotly-3D-Scatter mit Hover-Bild. |
| `pca_analysis_visualization.py` | Sammlung von PCA-Plots: Scree, Rekonstruktionsfehler, 2D/3D-Scatter, Pairplot, KDE, Patch-RGB-Overlay. |
| `cluster_ct_scans.py` | Vergleich/Clustering (UMAP + PCA) real vs. synthetisch. |
| `robust_regions.py` | Robuste per-Winkel-Regionserkennung (Mandrel / Layer / Tab / Gehäuse) auf dem entrollten Polarbild. |

## Ausgaben

Die Analyse-Skripte erzeugen ihre Ergebnisse lokal in `results/` (die Unterordner
werden automatisch angelegt); diese Dateien sind **nicht** im Repo versioniert
(siehe `.gitignore`):

- `results/html/polar_pca_interactive_3d_<modell>.html` – interaktive 3D-Darstellung je Modell (PC1/PC2/PC3, Hover zeigt den markierten Patch)
- `results/images/polar_pca_cumulative_violin_<modell>.png` – Violin-Plot je Modell
- `results/images/polar_pca_cumulative_violin_all_models.png` – Modell-Vergleichsbild (alle Modelle, real vs. synthetisch je Modell)
- `results/images/..._2d.png` – 2D-Scatter (PC1/PC2) je Modell
- `results/excel/polar_pca_variance.xlsx` – **eine** Excel-Datei mit zwei Arbeitsmappen je Modell (`<modell>_real`, `<modell>_synth`)

Bei nur einem Modell entfällt der `_<modell>`-Zusatz und das Vergleichsbild.
Auch die Eingabebilder (z. B. `real.png`) werden nicht versioniert und müssen
beim Ausführen selbst angegeben werden.

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

```bash
# Mehrere Modelle gleichzeitig, beide Analysen (Standard), real kartesisch + synth polar
python polar_pca_eval.py \
    --models dinov2-small dinov2-large dinov3-large --hf_token hf_xxx \
    --real_image real_cart.png       --real_format cart \
    --synthetic_image fake_polar.png --synth_format polar

# Nur das Violin-Vergleichsbild, zwei Modelle, beide kartesisch
python polar_pca_eval.py --analysis violin --models dinov2-large dinov3-large \
    --real_image real.png --real_format cart \
    --synthetic_image fake.png --synth_format cart
```

Standardmäßig laufen beide Analysen (`--analysis violin excel`), sodass jedes
Modell nur einmal embeddet wird. Die Ergebnis-Ordner lassen sich über
`--results_dir` bzw. `--html_dir` / `--excel_dir` / `--image_dir` steuern.

## Foundation-Modelle & DINOv3-Zugang

Verfügbare Modelle (`--models`): `dinov2-small`, `dinov2-base`, `dinov2-large`,
`dinov2-giant`, `dinov3-large`, `rad-dino`. Die Patch-Größe (14, bzw. 16 bei
DINOv3) wird pro Modell automatisch behandelt.

`dinov3-large` ist ein **gated** Modell auf Hugging Face. Einmalig nötig:

1. Hugging-Face-Konto anlegen und auf der Modellseite
   `facebook/dinov3-vitl16-pretrain-lvd1689m` Zugriff beantragen (Lizenz
   akzeptieren; Freigabe kann etwas dauern).
2. Einen **Access-Token (Read)** erstellen (Settings → Access Tokens).
3. Token bereitstellen – eine der Varianten:
   - `--hf_token hf_xxx` direkt als Flag, **oder**
   - Umgebungsvariable `HF_TOKEN=hf_xxx`, **oder**
   - `huggingface-cli login` (Token wird dauerhaft gecacht).

Außerdem muss `transformers` aktuell genug für DINOv3 sein (ggf.
`pip install -U transformers huggingface_hub`).

## Hinweis

`polar_embed.py` nutzt einen lokalen Default-Pfad (`DEFAULT_POLAR`) zu einem
Schwester-Projekt der Masterarbeit. Für die Ausführung auf einem anderen Rechner
diesen Pfad anpassen bzw. die entsprechenden CLI-Argumente setzen.
