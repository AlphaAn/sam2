import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def analyze_hiera_features(predictor, image_path):
    """
    Extracts and visualizes the intermediate feature maps from the SAM2 Hiera encoder.
    """
    print(f"\n--- Analyzing Encoder Features for: {image_path} ---")
    
    # 1. Load and process the image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # This triggers the Hiera encoder forward pass
    predictor.set_image(img_array)
    
    # 2. Extract the hidden data structures
    # SAM2 caches these internally after set_image() is called
    features = predictor._features
    
    if features is None:
        print("Error: Features not found. Ensure predictor.set_image() ran successfully.")
        return

    image_embed = features["image_embed"]
    high_res_feats = features["high_res_feats"]
    
    print("\n[Data Structures] Hiera Encoder Output Shapes:")
    for i, feat in enumerate(high_res_feats):
        print(f"  High-Res Feature Map {i+1}: {feat.shape} (Batch, Channels, Height, Width)")
    print(f"  Final Image Embedding:   {image_embed.shape} (Batch, Channels, Height, Width)")

    # 3. Visualization setup
    # We will plot the original image + each high-res map + the final embedding
    num_plots = 1 + len(high_res_feats) + 1 
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    
    # Plot Original Image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 4. Helper function to process and plot a tensor as a heatmap
    def plot_heatmap(ax, tensor, title):
        # Remove batch dimension: (1, C, H, W) -> (C, H, W)
        tensor = tensor.squeeze(0) 
        
        # Calculate the mean across the channel dimension (C) to get a 2D activation map (H, W)
        heatmap = torch.mean(tensor.float(), dim=0).cpu().numpy()
        
        # Normalize the heatmap to [0, 1] for better visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Plot using a colormap (viridis is standard for heatmaps)
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5. Plot the feature maps
    for i, feat in enumerate(high_res_feats):
        plot_heatmap(axes[i+1], feat, f"High-Res Map {i+1}\n{tuple(feat.shape[-2:])}")
        
    plot_heatmap(axes[-1], image_embed, f"Final Embedding\n{tuple(image_embed.shape[-2:])}")

    plt.tight_layout()
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and visualize intermediate feature maps from the SAM2 Hiera encoder.")
    
    # Required positional argument
    parser.add_argument("image", type=str, help="Path to the input image (e.g., images/groceries.jpg)")
    
    # Optional arguments with defaults
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", 
                        help="Path to the SAM2 model config")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt", 
                        help="Path to the SAM2 model checkpoint")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"], 
                        help="Device to run inference on. Auto-detects if not specified.")
    
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Initialize model
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    
    # Run inference and visualization
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            analyze_hiera_features(image_predictor, args.image)
    else:
        with torch.inference_mode():
            analyze_hiera_features(image_predictor, args.image)