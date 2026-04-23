import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_points_from_user(img_array):
    """
    Opens an OpenCV window to let the user click points on the image.
    Returns the coordinates and labels as numpy arrays.
    """
    print("\n--- Interactive Point Selection ---")
    print("Instructions:")
    print(" - LEFT CLICK to add a positive point (foreground).")
    print(" - RIGHT CLICK to add a negative point (background/exclude).")
    print(" - Press 'ENTER' or 'q' when you are finished selecting points.")
    
    # OpenCV expects BGR format for display
    display_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    points = []
    labels = []
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1) # 1 indicates a positive/foreground prompt
            
            # Draw a green dot at the clicked location
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1) 
            cv2.putText(display_img, "Positive", (x + 8, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("SAM2 Interactive Prompt", display_img)
            print(f"  [+] Added POSITIVE point at [{x}, {y}]")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0) # 0 indicates a negative/background prompt
            
            # Draw a red dot at the clicked location
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1) 
            cv2.putText(display_img, "Negative", (x + 8, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("SAM2 Interactive Prompt", display_img)
            print(f"  [-] Added NEGATIVE point at [{x}, {y}]")

    cv2.namedWindow("SAM2 Interactive Prompt")
    cv2.imshow("SAM2 Interactive Prompt", display_img)
    cv2.setMouseCallback("SAM2 Interactive Prompt", click_event)
    
    # Wait until the user presses 'Enter' (13) or 'q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    
    if not points:
        print("No points selected. Exiting.")
        exit()
        
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

def analyze_mask_decoder(predictor, image_path):
    print(f"\n--- Task 3: Mask Decoder Output Analysis ---")
    
    # 1. Load and set the image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    predictor.set_image(img_array)
    
    # 2. Get interactive prompts from the user
    input_points, input_labels = get_points_from_user(img_array)
    
    print(f"Prompting model with {len(input_points)} points...")

    # 3. Generate predictions
    # multimask_output=True forces the decoder to resolve ambiguity by offering 3 scales
    masks, iou_predictions, low_res_masks = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # 4. Parse and Analyze the Output Tensors
    print("\n[Data Structures] Mask Decoder Output Shapes:")
    print(f"  Final Masks:       {masks.shape}      (Output_Count, Height, Width) - Boolean Arrays")
    print(f"  IoU Predictions:   {iou_predictions.shape}            (Output_Count,) - Float Confidence Scores")
    print(f"  Low Res Masks:     {low_res_masks.shape}  (Output_Count, 256, 256) - Raw Logits")
    
    # Sort masks by their predicted IoU score to find the most "reliable" one
    sorted_indices = np.argsort(iou_predictions)[::-1]
    
    # 5. Visualization Setup
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot Original Image with the prompt points
    axes[0].imshow(img_array)
    
    # Plot positive (green) and negative (red) points separately for clarity
    pos_points = input_points[input_labels == 1]
    neg_points = input_points[input_labels == 0]
    
    if len(pos_points) > 0:
        axes[0].scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='*', s=150, edgecolor='black', label="Positive")
    if len(neg_points) > 0:
        axes[0].scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='X', s=150, edgecolor='black', label="Negative")
        
    axes[0].set_title(f"Original Image\n({len(input_points)} Prompts)")
    axes[0].axis('off')
    if len(input_points) > 0:
        axes[0].legend(loc='lower right')

    # Plot the 3 mask options, sorted by confidence
    for i, mask_idx in enumerate(sorted_indices):
        score = iou_predictions[mask_idx]
        
        # Force the mask into a strict boolean format
        mask = masks[mask_idx].astype(bool)
        
        # Create a color overlay for the mask
        color_mask = np.zeros((*mask.shape, 4))
        color_mask[mask] = [0.1, 0.5, 0.8, 0.6]  # Blue-ish overlay with 60% opacity
        
        axes[i+1].imshow(img_array)
        axes[i+1].imshow(color_mask)
        axes[i+1].set_title(f"Rank {i+1} Mask\nPredicted IoU: {score:.3f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SAM2 Mask Decoder outputs using interactive point prompts.")
    
    # Required positional argument
    parser.add_argument("image", type=str, help="Path to the input image (e.g., images/cars.jpg)")
    
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
    
    # Run inference handling autocast safely depending on the device
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            analyze_mask_decoder(image_predictor, args.image)
    else:
        with torch.inference_mode():
            analyze_mask_decoder(image_predictor, args.image)