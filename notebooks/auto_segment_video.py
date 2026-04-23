import argparse
import torch
import numpy as np
import os
import cv2
from PIL import Image

# SAM2 Imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def auto_segment_and_track_video(input_path, model_cfg, checkpoint_path, device):
    """
    Step 1: Automatically detects all objects in the first frame.
    Step 2: Tracks every detected object through the rest of the video.
    """
    print("\n--- Phase 1: Automatic Object Detection & Video Tracking ---")
    
    print("Loading SAM2 Base Model and Video Predictor...")
    # Load the base model for image auto-segmentation
    sam2_base = build_sam2(model_cfg, checkpoint_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_base)
    
    # Load the video predictor for temporal tracking
    video_predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    
    # 1. Extract the first frame to run auto-segmentation
    print(f"Reading Frame 0 from {input_path}...")
    is_file = os.path.isfile(input_path)
    if is_file:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {input_path}")
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to read the first frame of the video.")
        # Convert BGR to RGB for SAM2
        first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame_names = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not frame_names:
            raise ValueError(f"No valid image frames found in {input_path}")
        first_frame_path = os.path.join(input_path, frame_names[0])
        first_frame_rgb = np.array(Image.open(first_frame_path).convert("RGB"))
    
    # 2. Auto-Segment Frame 0
    print("Auto-detecting objects in Frame 0 (This may take a moment)...")
    initial_masks = mask_generator.generate(first_frame_rgb)
    
    # Optional: Sort masks by area to keep ordering consistent (largest to smallest)
    initial_masks = sorted(initial_masks, key=(lambda x: x['area']), reverse=True)
    print(f"Found {len(initial_masks)} distinct objects.")
    
    # 3. Initialize Video Tracking State
    print("Initializing video state (offloading memory to CPU to save VRAM)...")
    inference_state = video_predictor.init_state(
        video_path=input_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    )
    
    # 4. Load the auto-detected masks into the video predictor
    print("Loading objects into the tracking engine...")
    for obj_id, ann in enumerate(initial_masks, start=1):
        mask = ann['segmentation'] # Boolean numpy array (H, W)
        _, _, _ = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=mask,
        )
    
    # 5. Propagate through the whole video
    print("Propagating masks through the video (Tracking multiple objects takes time)...")
    video_segments = {} 
    
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx % 50 == 0:
            print(f"  Processed frame {out_frame_idx}...")
            
    return video_segments

def save_auto_tracking_results(video_segments, input_path, output_video_path="test/auto_tracked_output.mp4", save_frames=True):
    """
    Step 3: Renders the mathematical masks as colored overlays and saves the final MP4.
    """
    if not video_segments:
        print("No video segments generated. Exiting render phase.")
        return
        
    print(f"\n--- Phase 2: Rendering Final Output to {output_video_path} ---")
    
    # Setup Output Directories
    out_dir = os.path.dirname(output_video_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    frames_dir = None
    if save_frames:
        base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        frames_dir = os.path.join(out_dir, f"{base_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  -> Individual frames will be saved to: {frames_dir}/")
    
    # Determine input type and setup VideoWriter
    is_file = os.path.isfile(input_path)
    if is_file:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_names = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        first_frame = cv2.imread(os.path.join(input_path, frame_names[0]))
        height, width, _ = first_frame.shape
        fps = 30.0 # Default fallback
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    object_colors = {}
    frame_idx = 0
    
    while True:
        if is_file:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx >= len(frame_names):
                break
            frame = cv2.imread(os.path.join(input_path, frame_names[frame_idx]))
            if frame is None:
                break
            
        # Apply masks if SAM2 tracked objects on this frame
        if frame_idx in video_segments:
            objects_dict = video_segments[frame_idx]
            
            for obj_id, mask in objects_dict.items():
                if obj_id not in object_colors:
                    # Assign random distinct color
                    object_colors[obj_id] = np.random.randint(0, 255, (3,)).tolist()
                    
                color = object_colors[obj_id]
                mask_2d = mask.squeeze() 
                
                # Ensure mask is valid before applying overlay
                if np.any(mask_2d):
                    color_overlay = np.zeros_like(frame)
                    color_overlay[mask_2d] = color

                    roi_frame = frame[mask_2d]
                    roi_overlay = color_overlay[mask_2d]

                    if roi_frame.size > 0:
                        blended = cv2.addWeighted(roi_frame, 0.5, roi_overlay, 0.5, 0)
                        frame[mask_2d] = blended

                    # Draw label only if mask exists
                    y_coords, x_coords = np.where(mask_2d)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        min_x, min_y = np.min(x_coords), np.min(y_coords)
                        cv2.putText(frame, f"ID:{obj_id}", (min_x, max(15, min_y - 5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Write MP4
        out.write(frame)
        
        # Write independent frame
        if save_frames:
            frame_filename = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
        frame_idx += 1
        
    if is_file:
        cap.release()
    out.release()
    print(f"  -> Video rendered successfully!")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Automatic Multi-Object Video Tracking with SAM2")
    
    parser.add_argument("input_path", type=str, help="Path to the input MP4 file OR folder of raw frames")
    
    parser.add_argument("--output", type=str, default=None, help="Path to save the output MP4 file")

    parser.add_argument("--no_frames", action="store_true", 
                        help="Pass this flag if you ONLY want the MP4 and DO NOT want to save individual frames")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", 
                        help="Path to the SAM2 model config")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt", 
                        help="Path to the SAM2 model checkpoint")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"], 
                        help="Device to run inference on. Auto-detects if not specified.")
    
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input_path.rstrip("/\\"))
    
        if os.path.isfile(args.input_path):
            args.output = f"{base}_output.mp4"
        else:
            folder_name = os.path.basename(base)
            args.output = os.path.join(base, f"{folder_name}_output.mp4")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Run SAM2 inference pipeline
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracking_results = auto_segment_and_track_video(
                args.input_path, args.config, args.checkpoint, device
            )
    else:
        with torch.inference_mode():
            tracking_results = auto_segment_and_track_video(
                args.input_path, args.config, args.checkpoint, device
            )
            
    # Render MP4 and save frames
    save_frames_flag = not args.no_frames
    save_auto_tracking_results(tracking_results, args.input_path, args.output, save_frames=save_frames_flag)