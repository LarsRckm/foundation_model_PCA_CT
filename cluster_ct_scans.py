import argparse
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cosine, euclidean

def load_image(image_path):
    # Load image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    return img

def main(args):
    print("Loading DINOv2 model...")
    # Using facebook/dinov2-small as it is fast and robust
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')
    model.eval()

    if not os.path.exists(args.real_image) or not os.path.exists(args.synthetic_image):
        print(f"Error: Could not find image files.")
        print(f"Please ensure both '{args.real_image}' and '{args.synthetic_image}' exist in the directory.")
        return

    print("Loading images...")
    img_real = load_image(args.real_image)
    img_synth = load_image(args.synthetic_image)

    # Process Images
    inputs_real = processor(images=img_real, return_tensors="pt")
    inputs_synth = processor(images=img_synth, return_tensors="pt")
    
    # Get grid sizes (patch size for DINOv2 is 14)
    patch_size = 14
    h_real, w_real = inputs_real['pixel_values'].shape[2], inputs_real['pixel_values'].shape[3]
    h_synth, w_synth = inputs_synth['pixel_values'].shape[2], inputs_synth['pixel_values'].shape[3]
    
    grid_h_real, grid_w_real = h_real // patch_size, w_real // patch_size
    grid_h_synth, grid_w_synth = h_synth // patch_size, w_synth // patch_size

    print("Extracting features...")
    with torch.no_grad():
        outputs_real = model(**inputs_real)
        outputs_synth = model(**inputs_synth)

    # DINOv2 returns last_hidden_state of shape (batch_size, sequence_length, hidden_size)
    # The first token is the CLS token, the rest are patch tokens.
    patch_embeddings_real = outputs_real.last_hidden_state[0, 1:, :].numpy()
    patch_embeddings_synth = outputs_synth.last_hidden_state[0, 1:, :].numpy()

    # Verify grid sizes match the number of tokens
    if grid_h_real * grid_w_real != patch_embeddings_real.shape[0]:
        print("Warning: Real image grid size does not match token count.")
    if grid_h_synth * grid_w_synth != patch_embeddings_synth.shape[0]:
        print("Warning: Synthetic image grid size does not match token count.")

    # Global CLS embeddings for simple image-to-image distance
    cls_real = outputs_real.last_hidden_state[0, 0, :].numpy()
    cls_synth = outputs_synth.last_hidden_state[0, 0, :].numpy()

    print("\n--- Global Image Distance ---")
    print(f"Cosine Distance (CLS tokens): {cosine(cls_real, cls_synth):.4f}")
    print(f"Euclidean Distance (CLS tokens): {euclidean(cls_real, cls_synth):.4f}")

    # Combine patch embeddings
    X = np.vstack((patch_embeddings_real, patch_embeddings_synth))
    
    # Create labels: 0 for Real, 1 for Synthetic
    labels = np.concatenate([
        np.zeros(patch_embeddings_real.shape[0]), 
        np.ones(patch_embeddings_synth.shape[0])
    ])

    # 1. Dimensionality Reduction for 2D Scatter Plot
    print("\nPerforming dimensionality reduction for 2D Scatter Plot...")
    if args.reduction == 'umap':
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer_2d = PCA(n_components=2, random_state=42)
        
    X_reduced_2d = reducer_2d.fit_transform(X)

    # 2. PCA for Spatial RGB Mapping
    # We use PCA with 3 components to map the latent space to RGB channels (Red, Green, Blue)
    print("Performing PCA for Spatial RGB Mapping...")
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X)
    
    # Normalize the 3 PCA components to [0, 1] range so they can be plotted as colors
    X_pca_3d_norm = (X_pca_3d - X_pca_3d.min(axis=0)) / (X_pca_3d.max(axis=0) - X_pca_3d.min(axis=0))

    # Split back into real and synthetic
    pca_rgb_real = X_pca_3d_norm[:patch_embeddings_real.shape[0]]
    pca_rgb_synth = X_pca_3d_norm[patch_embeddings_real.shape[0]:]

    # Reshape the 1D list of colors back into the 2D grid of the original image
    pca_img_real = pca_rgb_real.reshape(grid_h_real, grid_w_real, 3)
    pca_img_synth = pca_rgb_synth.reshape(grid_h_synth, grid_w_synth, 3)

    # 3. Visualization
    print("Generating visualization...")
    plt.figure(figsize=(18, 5))

    # Plot 1: 2D Scatter Plot
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=15)
    plt.title(f"{args.reduction.upper()} Scatter Plot")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(0.0), markersize=8, label='Real Patches'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(1.0), markersize=8, label='Synthetic Patches')]
    plt.legend(handles=legend_elements)

    # Plot 2: Spatial PCA Heatmap (Real)
    plt.subplot(1, 3, 2)
    plt.imshow(pca_img_real)
    plt.title("Real Scan - PCA Spatial Map")
    plt.axis('off')

    # Plot 3: Spatial PCA Heatmap (Synthetic)
    plt.subplot(1, 3, 3)
    plt.imshow(pca_img_synth)
    plt.title("Synthetic Scan - PCA Spatial Map")
    plt.axis('off')

    plt.tight_layout()
    output_filename = "latent_space_visualization.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nSaved visualization to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster and compare battery CT scans in DINOv2 latent space.")
    parser.add_argument("--real_image", type=str, default="real.png", help="Path to the real CT scan image (PNG)")
    parser.add_argument("--synthetic_image", type=str, default="fake.png", help="Path to the synthetic CT scan image (PNG)")
    parser.add_argument("--reduction", type=str, choices=["pca", "umap"], default="pca", help="Dimensionality reduction technique (umap or pca)")
    args = parser.parse_args()
    main(args)
