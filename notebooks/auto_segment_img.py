import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def auto_segment_image(mask_generator, image_path):
    print(f"\n--- Running Automatic Whole-Image Segmentation ---")
    
    # 1. Load the image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # 2. Generate all masks
    print(f"Analyzing {image_path}...")
    print("Generating masks across the entire image (this may take a moment)...")
    masks = mask_generator.generate(img_array)
    
    print(f"Successfully generated {len(masks)} individual masks!")
    
    # 3. Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(img_array)
    
    # Iterate through every detected mask and overlay it with a random color
    if len(masks) > 0:
        # Sort masks by area so smaller objects are drawn on top of larger ones
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        for ann in sorted_masks:
            m = ann['segmentation'] # This is a boolean array
            color_mask = np.concatenate([np.random.random(3), [0.5]]) # Random RGB + 50% Alpha
            
            # Create an empty overlay image, then apply the color only where the mask is True
            img_overlay = np.zeros((m.shape[0], m.shape[1], 4))
            img_overlay[m] = color_mask
            plt.imshow(img_overlay)

    plt.axis('off')
    plt.title(f"Auto-Segmented {len(masks)} Objects")
    plt.tight_layout()
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks using SAM2 Automatic Mask Generation (Whole-Image Segmentation).")
    
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
    print("Loading SAM2 model...")
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    
    # Initialize the Automatic Generator
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    
    # Run inference handling autocast safely depending on the device
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            auto_segment_image(mask_generator, args.image)
    else:
        with torch.inference_mode():
            auto_segment_image(mask_generator, args.image)