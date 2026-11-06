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
| `polar_pca_eval.py` | Haupteinstiegspunkt: real + synthetisches Bild → (optional) Kartesisch→Polar-Transform → Foundation-Model-Embeddings → wahlweise Violin- oder Excel+3D-Analyse. |
| `polar_embed.py` | Gemeinsame Helfer: Polar-Transform + Patch-Embedding-Extraktion; Modell-Registry (DINOv2 small/base/large/giant, DINOv3-large, rad-dino). |
| `pca_cumulative_violin.py` | Split-Violin-Plot der kumulativen erklärten Varianz (real vs. synthetisch). |
| `pca_excel_interactive.py` | Excel-Report (Varianz je Hauptkomponente) + interaktiver Plotly-3D-Scatter mit Hover-Bild. |
| `pca_analysis_visualization.py` | Sammlung von PCA-Plots: Scree, Rekonstruktionsfehler, 2D/3D-Scatter, Pairplot, KDE, Patch-RGB-Overlay. |
| `cluster_ct_scans.py` | Vergleich/Clustering (UMAP + PCA) real vs. synthetisch. |
| `robust_regions.py` | Robuste per-Winkel-Regionserkennung (Mandrel / Layer / Tab / Gehäuse) auf dem entrollten Polarbild. |

## Ausgaben (im Repo enthalten)

- `polar_pca_interactive_3d.html` – interaktive 3D-Darstellung (PC1/PC2/PC3, Hover zeigt den markierten Patch)
- `polar_pca_interactive_3d_2d.png`, `polar_pca_cumulative_violin.png`, `real.png`
- `polar_pca_variance.xlsx` – Varianz je Hauptkomponente

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

```bash
# Kumulative-Varianz-Violins, DINOv2-large, real kartesisch + synth polar
python polar_pca_eval.py --analysis violin --model dinov2-large \
    --real_image real_cart.png       --real_format cart \
    --synthetic_image fake_polar.png --synth_format polar

# Excel-Report + interaktives 3D-HTML, DINOv3, beide kartesisch
python polar_pca_eval.py --analysis excel --model dinov3-large \
    --real_image real.png --real_format cart \
    --synthetic_image fake.png --synth_format cart
```

## Hinweis

`polar_embed.py` nutzt einen lokalen Default-Pfad (`DEFAULT_POLAR`) zu einem
Schwester-Projekt der Masterarbeit. Für die Ausführung auf einem anderen Rechner
diesen Pfad anpassen bzw. die entsprechenden CLI-Argumente setzen.
