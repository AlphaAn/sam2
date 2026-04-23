import argparse
import torch
import numpy as np
import os
import cv2
from sam2.build_sam import build_sam2_video_predictor

# --- Global Variables for GUI ---
object_prompts = {} 
current_obj_id = 1
display_img = None

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to assign positive/negative points to objects."""
    global current_obj_id, display_img, object_prompts
    
    if current_obj_id not in object_prompts:
        object_prompts[current_obj_id] = {'points': [], 'labels': []}
        
    if event == cv2.EVENT_LBUTTONDOWN:
        object_prompts[current_obj_id]['points'].append([x, y])
        object_prompts[current_obj_id]['labels'].append(1)
        
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1) 
        cv2.putText(display_img, f"ID:{current_obj_id}", (x + 8, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("SAM2 Interactive Tracker", display_img)
        print(f"  [+] Added POSITIVE point to Object {current_obj_id} at [{x}, {y}]")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        object_prompts[current_obj_id]['points'].append([x, y])
        object_prompts[current_obj_id]['labels'].append(0)
        
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1) 
        cv2.putText(display_img, f"EXCLUDE", (x + 8, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("SAM2 Interactive Tracker", display_img)
        print(f"  [-] Added NEGATIVE point to Object {current_obj_id} at [{x}, {y}]")

def get_user_clicks(input_path):
    """Extracts the first frame (from MP4 or directory) and opens the GUI."""
    global display_img, object_prompts, current_obj_id
    
    object_prompts = {}
    current_obj_id = 1
    
    # Check if the input is a file or a directory
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {input_path}")
        ret, display_img = cap.read()
        cap.release() 
        if not ret:
            raise ValueError("Failed to read the first frame of the video.")
    elif os.path.isdir(input_path):
        frame_names = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not frame_names:
            raise ValueError(f"No image frames found in directory: {input_path}")
        display_img = cv2.imread(os.path.join(input_path, frame_names[0]))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    cv2.namedWindow("SAM2 Interactive Tracker")
    cv2.setMouseCallback("SAM2 Interactive Tracker", mouse_callback)
    
    print("\n" + "="*60)
    print(f"INTERACTIVE MULTI-POINT SELECTION ({'FILE' if os.path.isfile(input_path) else 'FOLDER'} MODE)")
    print(f"--> Currently selecting points for OBJECT {current_obj_id}")
    print("  * LEFT-CLICK: Add a point to INCLUDE in the object (Green)")
    print("  * RIGHT-CLICK: Add a point to EXCLUDE / Background (Red)")
    print("  * PRESS 'n': Move on to the NEXT object")
    print("  * PRESS 'q' or ENTER: Finish and start tracking")
    print("="*60 + "\n")
    
    cv2.imshow("SAM2 Interactive Tracker", display_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            if current_obj_id in object_prompts and len(object_prompts[current_obj_id]['points']) > 0:
                current_obj_id += 1
                print(f"\n--> Switched! Now selecting points for OBJECT {current_obj_id}")
            else:
                print("  Please add at least one point before switching to a new object.")
        elif key == ord('q') or key == 13: 
            break
            
    cv2.destroyAllWindows()
    
    clean_prompts = {k: v for k, v in object_prompts.items() if len(v['points']) > 0}
    return clean_prompts

def interactive_multi_track(input_path, prompts_dict, model_cfg, checkpoint_path, device):
    """Loads the video/frames into SAM2, applies the prompts, and tracks."""
    if not prompts_dict:
        print("No objects selected. Exiting.")
        return {}
    
    print("\nLoading SAM2 Video Predictor...")
    video_predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    
    print(f"Initializing video state for {input_path}...")
    inference_state = video_predictor.init_state(video_path=input_path)
    
    for obj_id, data in prompts_dict.items():
        points = np.array(data['points'], dtype=np.float32)
        labels = np.array(data['labels'], dtype=np.int32) 
        
        print(f"Loading Object {obj_id} into engine with {len(points)} points...")
        _, _, _ = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0, 
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
    
    print("Propagating masks through the video...")
    video_segments = {}
    
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx % 50 == 0:
            print(f"  Processed frame {out_frame_idx}...")
            
    return video_segments

def save_interactive_results(video_segments, input_path, output_video_path="test/tracked_output.mp4", save_frames=True):
    """Reads the original input, applies tracking masks, and saves MP4 + frames."""
    if not video_segments:
        return
        
    print(f"\n--- Rendering Final Output ---")
    
    # 1. Setup Output Directories
    out_dir = os.path.dirname(output_video_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    frames_dir = None
    if save_frames:
        base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        frames_dir = os.path.join(out_dir, f"{base_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  -> Individual frames will be saved to: {frames_dir}/")
    
    # 2. Determine input type and setup VideoWriter
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
        fps = 30.0 # Default fallback for image sequences
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    object_colors = {}
    frame_idx = 0
    
    while True:
        # Load the next frame based on input type
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
        
        # Write to the new MP4
        out.write(frame)
        
        # Write the individual frame to disk
        if save_frames:
            frame_filename = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
        frame_idx += 1
        
    if is_file:
        cap.release()
    out.release()
    print(f"  -> Video rendered successfully to: {output_video_path}")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Multi-Object Video Tracking with SAM2 (Supports MP4 & Folders)")
    
    # Accept either file or folder
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
    
    # 1. Run the interactive GUI
    user_prompts = get_user_clicks(args.input_path)
    
    # 2. Run SAM2 inference
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracking_results = interactive_multi_track(
                args.input_path, user_prompts, args.config, args.checkpoint, device
            )
    else:
        with torch.inference_mode():
            tracking_results = interactive_multi_track(
                args.input_path, user_prompts, args.config, args.checkpoint, device
            )
            
    # 3. Render MP4 and save frames
    save_frames_flag = not args.no_frames
    save_interactive_results(tracking_results, args.input_path, args.output, save_frames=save_frames_flag)