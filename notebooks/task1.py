import torch
import numpy as np
import gc
from PIL import Image
import os
from pathlib import Path

# SAM 2 API Imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def test_image_formats(predictor, image_paths):
    """
    Tests SAM2 compatibility with various image formats (.jpg, .png, .tiff).
    """
    print("--- Image Compatibility Test ---")
    for path in image_paths:
        if not os.path.exists(path):
            print(f"[Skip] File not found: {path}")
            continue
        try:
            # SAM2 expects an RGB numpy array
            img = Image.open(path).convert("RGB")
            img_array = np.array(img)
            
            # Predictor generates and stores the image embedding
            predictor.set_image(img_array)
            print(f"[Success] {path} | Shape: {img_array.shape} | Format: {img.format}")
        except Exception as e:
            print(f"[Failed] {path} | Error: {e}")
        finally:
            torch.cuda.empty_cache()

def test_video_formats(predictor, video_paths):
    """
    Tests SAM2 compatibility with video files vs raw frame directories.
    """
    print("\n--- Video Compatibility Test ---")
    for path in video_paths:
        if not os.path.exists(path):
            print(f"[Skip] Path not found: {path}")
            continue
        try:
            state = predictor.init_state(video_path=path)
            print(f"[Success] Initialized video state for: {path}")
        except Exception as e:
            print(f"[Failed] Video state initialization for {path} | Error: {e}")
        finally:
            torch.cuda.empty_cache()

def resolution_oom_stress_test(predictor, start_res=16384, step=16384):
    """
    Gradually increases image resolution through the SAM2 encoder until 
    an OOM error is triggered.
    """
    print(f"\n--- OOM Resolution Stress Test ({device}) ---")
    if device.type != "cuda":
        print("CUDA not detected. Skipping OOM stress test to avoid crashing system RAM.")
        return

    current_res = start_res
    while True:
        try:
            print(f"Testing spatial resolution: {current_res}x{current_res}...")
            # Generate a dummy RGB numpy array simulating an image
            dummy_image = np.random.randint(0, 255, (current_res, current_res, 3), dtype=np.uint8)
            
            # Process the image to generate embeddings
            predictor.set_image(dummy_image)
            
            print(f"  -> Passed {current_res}x{current_res}")
            current_res += step
            
            # Force memory cleanup before the next larger tensor
            predictor.reset_predictor()
            torch.cuda.empty_cache()
            gc.collect()
            
        except torch.OutOfMemoryError:
            print(f"!!! OOM Error triggered at {current_res}x{current_res} !!!")
            print(f"Maximum safe resolution on your hardware is roughly {current_res - step}x{current_res - step}.")
            
            # Final cleanup to leave GPU in a usable state
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as e:
            print(f"Stopped at {current_res}x{current_res} due to unexpected error: {e}")
            break

if __name__ == "__main__":
    # 1. Model Configuration
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 
    checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
    
    print("Loading SAM2 models...")
    print(f"Using device: {device}")
    try:
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        image_predictor = SAM2ImagePredictor(sam2_model)
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    except Exception as e:
        print(f"Model loading failed. Ensure checkpoints exist. Error: {e}")
        exit()

    # 2. Dynamically scan the test folder
    test_dir = Path("test/input")
    test_imgs = []
    test_vids = []

    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Directory '{test_dir}' not found. Please create it and add your media files.")
    else:
        print(f"Scanning '{test_dir}' directory...")
        image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.avif'}
        video_exts = {'.mp4', '.mov', '.avi', '.mkv'}

        for path in test_dir.iterdir():
            if path.is_file():
                ext = path.suffix.lower()
                if ext in image_exts:
                    test_imgs.append(str(path))
                elif ext in video_exts:
                    test_vids.append(str(path))
            elif path.is_dir():
                # SAM2 treats directories of JPEGs as a video sequence
                test_vids.append(str(path))

        print(f"Found {len(test_imgs)} images and {len(test_vids)} video sources (files/directories).")

    # 3. Run the implementation checks
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if test_imgs:
            test_image_formats(image_predictor, test_imgs)
        else:
            print("\n[Skip] No images found to test.")
            
        if test_vids:
            test_video_formats(video_predictor, test_vids)
        else:
            print("\n[Skip] No videos or frame directories found to test.")
        
        # Run the stress test
        resolution_oom_stress_test(image_predictor, start_res=16384, step=16384)