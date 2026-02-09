import os
import json
import cv2
import numpy as np
import glob
import shutil
import argparse
from tqdm import tqdm

def process_data(raw_root, new_root, k=15):
    """
    Process video data: 
    1. Extract ALL frames from videos.
    2. Sample frames based on task annotations (using round() for 'm').
    3. Copy annotation files.
    """
    # 0. Copy task_instruction.json
    raw_instruction_path = os.path.join(raw_root, "task_instruction.json")
    new_instruction_path = os.path.join(new_root, "task_instruction.json")
    
    # Ensure new_root exists
    os.makedirs(new_root, exist_ok=True)

    if os.path.exists(raw_instruction_path):
        shutil.copy(raw_instruction_path, new_instruction_path)
        print(f"Copied task_instruction.json to {new_instruction_path}")
    else:
        print(f"Warning: {raw_instruction_path} not found.")

    # Find all episode_xxx folders
    episode_dirs = sorted(glob.glob(os.path.join(raw_root, 'episode_*')))
    
    if not episode_dirs:
        print(f"No episode folders found in {raw_root}")
        return

    # Define camera names to process
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

    for ep_path in tqdm(episode_dirs, desc="Processing Episodes"):
        episode_name = os.path.basename(ep_path)
        out_ep_dir = os.path.join(new_root, episode_name)
        os.makedirs(out_ep_dir, exist_ok=True)
        
        # 1. Get physical video information first (needed for default logic)
        ref_video_path = os.path.join(ep_path, 'cam_high.mp4')
        if not os.path.exists(ref_video_path):
            print(f"Skipping {episode_name}: cam_high.mp4 not found.")
            continue
            
        cap = cv2.VideoCapture(ref_video_path)
        physical_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 2. Handle annotated_keyframes.json
        json_path = os.path.join(ep_path, 'annotated_keyframes.json')
        annotations = []
        
        calc_total_frames = 0 # Frame count used for 'k' calculation
        
        if os.path.exists(json_path):
            # Copy the annotation file
            out_json_path = os.path.join(out_ep_dir, 'annotated_keyframes.json')
            shutil.copy(json_path, out_json_path)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # Logic: total_frame = last task end - first task start
            if annotations:
                start_f = annotations[0]['start_frame_id']
                end_f = annotations[-1]['end_frame_id']
                calc_total_frames = end_f - start_f
            else:
                # Handle empty list in json
                 calc_total_frames = physical_video_frames
        else:
            # Handle missing json
            print(f"Warning: {episode_name} missing annotated_keyframes.json. Using default full-video range.")
            calc_total_frames = physical_video_frames
            # Create a default task covering the whole video
            annotations = [{
                "annotation": "default_task",
                "start_frame_id": 0,
                "end_frame_id": physical_video_frames
            }]

        # 3. Calculate Sample IDs
        # Calculate n (total samples needed based on the calculated logic)
        if calc_total_frames <= 0: 
            calc_total_frames = 1 # Avoid division by zero

        n = calc_total_frames / k
        
        subtask_count = len(annotations)
        
        # Calculate m (samples per subtask)
        # Changed from floor to round as requested
        if subtask_count > 0:
            m = int(round(n / subtask_count))
        else:
            m = 0

        all_sample_ids = set()

        for task in annotations:
            start_id = task['start_frame_id']
            end_id = task['end_frame_id']
            
            # Logic: Sample m-1 frames equally spaced, plus start and end
            num_points = (m - 1) + 2
            if num_points < 2: 
                num_points = 2 

            # Generate indices
            indices = np.linspace(start_id, end_id, num_points)
            indices = np.round(indices).astype(int)
            
            # Filter valid indices (within physical video limits)
            valid_indices = [idx for idx in indices if 0 <= idx < physical_video_frames]
            all_sample_ids.update(valid_indices)

        sorted_sample_ids = sorted(list(all_sample_ids))
        sample_id_strs = [f"{idx:06d}" for idx in sorted_sample_ids]

        # Save sample_frames.json
        out_sample_json_path = os.path.join(out_ep_dir, 'sample_frames.json')
        with open(out_sample_json_path, 'w') as f:
            json.dump(sample_id_strs, f, indent=4)

        # 4. Extract ALL frames
        for cam in camera_names:
            video_file = os.path.join(ep_path, f"{cam}.mp4")
            if not os.path.exists(video_file):
                continue
            
            img_out_dir = os.path.join(out_ep_dir, cam)
            os.makedirs(img_out_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_file)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save every frame
                file_name = f"frame_{frame_idx:06d}.jpg"
                save_path = os.path.join(img_out_dir, file_name)
                cv2.imwrite(save_path, frame)
                
                frame_idx += 1

            cap.release()

def main():
    parser = argparse.ArgumentParser(description="Process video episodes and sample frames.")
    parser.add_argument("--raw_dir", type=str, default="example_raw_data", help="Path to the raw data directory containing episodes")
    parser.add_argument("--cvt_dir", type=str, default="train_data", help="Path to the output directory for processed data")
    parser.add_argument("--sample_factor", type=int, default=20, help="Sampling factor (avg_num_of_sample_frames = total_frames / k)")

    args = parser.parse_args()

    # Print configuration
    print("-" * 30)
    print("Configuration:")
    print("-" * 30)
    print(f"  Raw Data Directory: {args.raw_dir}")
    print(f"  Output Directory  : {args.cvt_dir}")
    print(f"  K Factor          : {args.sample_factor}")
    print("-" * 30)

    # Run processing
    process_data(args.raw_dir, args.cvt_dir, args.sample_factor)
    print("Processing complete!")

if __name__ == "__main__":
    main()
