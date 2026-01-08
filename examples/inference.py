import os
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# VLLM & Transformers
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Global Environment Settings
os.environ['LOCAL_RANK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# -----------------------------
# Configuration & Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""

# -----------------------------
# File & Video Utilities
# -----------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_pngs_sorted(dir_path: Path) -> List[Path]:
    """Return lexicographically sorted .png files (case-insensitive) under dir_path."""
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])

def get_frame_count(path: Path) -> Tuple[str, int]:
    """
    Detects if path is a video file or a directory of images.
    Returns: (source_type, frame_count). source_type is 'dir' or 'video'.
    """
    if path.is_dir():
        files = list_pngs_sorted(path)
        if not files:
            raise RuntimeError(f"No PNG frames found in directory: {path}")
        return "dir", len(files)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            raise RuntimeError(f"Invalid frame count from video: {path}")
        return "video", n

def make_sample_indices_by_interval(num_frames: int, interval: int) -> List[int]:
    """
    Generate indices based on a fixed frame interval.
    Always includes the first frame (0) and ensures the very last frame (num_frames-1) is included.
    """
    if num_frames < 1:
        return []
    
    # Generate base steps: 0, interval, 2*interval, ...
    indices = list(range(0, num_frames, interval))
    
    # Ensure the absolute last frame is included to cover the full episode
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)
        
    return indices

def save_frames(src_path: Path, out_dir: Path, indices: List[int], src_type: str) -> None:
    """
    Extracts and saves specific frames from a video or copies them from a directory.
    Output format: frame_{:06d}.png
    """
    ensure_dir(out_dir)
    
    if src_type == "video":
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = cap.read() # Retry once
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {idx} from {src_path}")
                continue
                
            out_path = out_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cap.release()
        
    elif src_type == "dir":
        files = list_pngs_sorted(src_path)
        n = len(files)
        for idx in indices:
            if not (0 <= idx < n):
                continue
            src = files[idx]
            dst = out_dir / f"frame_{idx:06d}.png"
            shutil.copyfile(src, dst)

def build_samples_json(
    run_root: Path, 
    task: str, 
    indices: List[int], 
    ref_end_path: str,
    mode: str = "incremental"
) -> List[Dict]:
    """
    Constructs the list of samples for VLLM inference.
    
    Args:
        mode: "incremental", "forward", or "backward"
        ref_end_path: Absolute path to the Goal Image.
    """
    timestamp_name = run_root.name
    cache_root = run_root / ".cache"
    items = []
    
    if len(indices) < 2:
        return items

    # For Forward mode, we need the Start Frame (Index 0)
    # For Incremental, we use Index k
    # For Backward, we use Goal Image
    
    for k in range(len(indices) - 1):
        af = indices[k+1] # The "After" frame (Current Step)
        
        # Define "Before" images based on mode
        bf_images = []
        bf_id_str = ""

        if mode == "incremental":
            bf = indices[k]
            bf_id_str = f"bf_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            # Compare Start(0) -> Current(af)
            bf = indices[0]
            bf_id_str = f"start_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            # Compare Goal -> Current(af)
            # Use ref_end_path for all 3 views if wrist goals aren't explicit
            bf_id_str = "goal"
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Define "After" images (Always the current step)
        af_images = [
            str(cache_root / "cam_high"        / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist"  / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]

        items.append({
            "id": f"step-{timestamp_name}-{k:04d}-{bf_id_str}-af_{af:06d}",
            "task": task,
            "image": [
                str(cache_root / "cam_high"        / f"frame_{0:06d}.png"),        # 1. Ref Start
                ref_end_path,                                                      # 2. Ref End
                bf_images[0],                                                      # 3. Before High
                bf_images[1],                                                      # 4. Before Left
                bf_images[2],                                                      # 5. Before Right
                af_images[0],                                                      # 6. After High
                af_images[1],                                                      # 7. After Left
                af_images[2],                                                      # 8. After Right
            ]
        })
    return items

# -----------------------------
# Visualization
# -----------------------------

def plot_video_reward(episode_root: Path):
    """
    Generates a visualization video (reward_vis.mp4).
    Feature: Uses 'symlog' (Symmetric Log) scale for the Hop plot.
    """
    pred_path = episode_root / "pred_vllm.json"
    if not pred_path.exists():
        print(f"[WARN] No prediction file found at {pred_path}, skipping visualization.")
        return

    with open(pred_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        return

    def get_fid(p):
        m = re.search(r"frame_(\d+)\.png", os.path.basename(p))
        return int(m.group(1)) if m else 0

    W_panel, H_panel = 384, 288
    H_plot = 260
    W_total = 3 * W_panel
    H_total = H_panel + H_plot
    
    out_path = episode_root / "reward_vis.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, 4.0, (W_total, H_total))
    if not writer.isOpened():
        print("[WARN] 'avc1' codec failed, falling back to 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 4.0, (W_total, H_total))
    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {out_path}")
        return

    xs_step, ys_hop, c_hop = [], [], []
    xs_frame, ys_prog, c_prog = [], [], []

    def draw_plots():
        dpi = 100
        fig = plt.figure(figsize=(W_total / dpi, H_plot / dpi), dpi=dpi)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot 1: Step vs Hop (SymLog)
        if xs_step:
            if len(xs_step) > 1:
                ax1.plot(xs_step, ys_hop, "-", color="#999999", alpha=0.6)
            ax1.scatter(xs_step, ys_hop, c=c_hop, s=36, edgecolors="k", linewidths=0.5)
        
        ax1.set_yscale('symlog', linthresh=0.05)
        ax1.set_ylim(-1.5, 1.5) 
        ticks = [-1.0, -0.1, 0, 0.1, 1.0]
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(["-100%", "-10%", "0", "10%", "100%"])
        
        max_step = len(items) + 1
        ax1.set_xlim(left=0, right=max(5, max_step)) 
        ax1.set_title("Instant Hop/Reward (SymLog)")
        ax1.grid(True, which="major", linestyle="-", alpha=0.4)

        # Plot 2: Frame vs Progress (Linear)
        if xs_frame:
            if len(xs_frame) > 1:
                ax2.plot(xs_frame, ys_prog, "-", color="#999999", alpha=0.6)
            ax2.scatter(xs_frame, ys_prog, c=c_prog, s=36, edgecolors="k", linewidths=0.5)
        
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Accumulated Progress")
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout(pad=1.0)
        
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.resize(img, (W_total, H_plot), interpolation=cv2.INTER_AREA)

    try:
        # --- Initial Frame ---
        first = items[0]
        # In Backward/Forward mode, image[2,3,4] (Before) might be Goal or Start.
        # This visualization logic assumes standard display.
        imgs_top = [cv2.imread(first["image"][i]) for i in [3, 2, 4]]
        if any(im is None for im in imgs_top):
            writer.release()
            return

        top_row = np.hstack([cv2.resize(im, (W_panel, H_panel)) for im in imgs_top])
        
        # Frame 0 is Ref Start
        xs_step.append(0)
        ys_hop.append(0.0)
        c_hop.append("#808080")
        
        xs_frame.append(0)
        ys_prog.append(0.0)
        c_prog.append("#00A000")
        
        bottom_plot = draw_plots()
        frame_comp = np.vstack([top_row, bottom_plot])
        
        if frame_comp.shape[:2] != (H_total, W_total):
            frame_comp = cv2.resize(frame_comp, (W_total, H_total))

        cv2.putText(frame_comp, "Start", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        writer.write(frame_comp)

        # --- Loop ---
        prev_prog = 0.0
        cum_frame = 0
        
        for i, item in enumerate(items):
            imgs_top = [cv2.imread(item["image"][idx]) for idx in [6, 5, 7]]
            if any(im is None for im in imgs_top): 
                continue

            top_row = np.hstack([cv2.resize(im, (W_panel, H_panel)) for im in imgs_top])

            hop = float(item.get("hop", 0))
            prog = float(item.get("progress", 0))
            
            step_id = i + 1
            # Simple frame counter approximation
            cum_frame += 10 # Just for viz x-axis spacing if real parsing fails
            try:
                # Try real parsing
                fid_curr = get_fid(item["image"][5])
                # We want absolute frame id on x-axis
                cum_frame = fid_curr
            except:  # noqa: E722
                pass
            
            xs_step.append(step_id)
            ys_hop.append(hop)
            c_hop.append("#00A000" if hop > 0 else ("#CC0000" if hop < 0 else "#808080"))

            xs_frame.append(cum_frame)
            ys_prog.append(prog)
            c_prog.append("#00A000" if prog >= prev_prog else "#CC0000")
            prev_prog = prog

            bottom_plot = draw_plots()
            frame_comp = np.vstack([top_row, bottom_plot])
            
            if frame_comp.shape[:2] != (H_total, W_total):
                frame_comp = cv2.resize(frame_comp, (W_total, H_total))
            
            info_txt = f"Step={step_id} Hop={hop:.2f} Prog={prog:.2f}"
            cv2.putText(frame_comp, info_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            writer.write(frame_comp)

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        writer.release()
        print(f"Video saved to: {out_path}")

# -----------------------------
# Main Inference Class
# -----------------------------

class GRMInference:
    def __init__(self, model_path: str, max_image_num=8, min_pixels=12544, max_pixels=76800):
        print(f"Loading model from {model_path} ...")

        self.model = LLM(
            model=model_path,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1, top_p=0.9, top_k=50, max_tokens=1024
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels

    def inference_batch(self, batch_data: List[Dict]) -> List[Dict]:
        prompts = []
        for item in batch_data:
            images = [Image.open(p).convert("RGB") for p in item["image"]]
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[0]},
                    {"type": "image"}, # Ref Start
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[1]},
                    {"type": "image"}, # Ref End
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[2]},
                    {"type": "image"}, # BF High
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[3]},
                    {"type": "image"}, # BF Left
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[4]},
                    {"type": "image"}, # BF Right
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[5]},
                    {"type": "image"}, # AF High
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[6]},
                    {"type": "image"}, # AF Left
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[7]},
                    {"type": "image"}, # AF Right
                    {"type": "text", "text": SYSTEM_PROMPT.format(task=item["task"]).split("<image>")[8]},
                ]
            }]
            
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": images}
            })

        outputs = self.model.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        
        results = []
        for orig_item, out in zip(batch_data, outputs):
            res_item = orig_item.copy()
            res_item["pred"] = out.outputs[0].text
            results.append(res_item)
        return results

    def run_pipeline(
        self,
        cam_high_path: str,
        cam_left_path: str,
        cam_right_path: str,
        out_root: str,
        task: str,
        frame_interval: int = 10,
        batch_size: int = 1,
        goal_image: Optional[str] = None,
        eval_mode: str = "incremental",
        visualize: bool = False
    ) -> str:
        """
        Main entry point.
        Args:
            eval_mode: 'incremental', 'forward', or 'backward'
        """
        valid_modes = ["incremental", "forward", "backward"]
        if eval_mode not in valid_modes:
            raise ValueError(f"Invalid eval_mode '{eval_mode}'. Must be one of {valid_modes}")

        out_root = Path(out_root)
        ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        run_root = out_root / ts
        cache_root = run_root / ".cache"
        
        cam_dirs = {
            "cam_high": cache_root / "cam_high",
            "cam_left_wrist": cache_root / "cam_left_wrist",
            "cam_right_wrist": cache_root / "cam_right_wrist",
        }
        for d in cam_dirs.values():
            ensure_dir(d)

        paths = [Path(cam_high_path), Path(cam_left_path), Path(cam_right_path)]
        types_counts = [get_frame_count(p) for p in paths]
        
        counts = [tc[1] for tc in types_counts]
        if len(set(counts)) != 1:
            raise ValueError(f"Frame count mismatch among cameras: {counts}")
        total_frames = counts[0]
        
        indices = make_sample_indices_by_interval(total_frames, frame_interval)
        print(f"Frames: {total_frames}, Int: {frame_interval}, Mode: {eval_mode}, Indices: {len(indices)}")

        for p, key, (stype, _) in zip(paths, cam_dirs.keys(), types_counts):
            save_frames(p, cam_dirs[key], indices, stype)

        # Handle Goal Image / Ref End
        if goal_image is not None and os.path.exists(goal_image):
            ref_end_path = cache_root / "ref_end.png"
            shutil.copy(goal_image, ref_end_path)
            ref_end_path_str = str(ref_end_path)
            print(f"Using Goal Image: {goal_image}")
        else:
            ref_end_path_str = str(cam_dirs["cam_high"] / f"frame_{total_frames-1:06d}.png")
            print("No Goal Image provided. Using last frame.")

        # Build Samples based on Mode
        samples = build_samples_json(run_root, task, indices, ref_end_path_str, mode=eval_mode)
        json_path = run_root / "sample.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"Running inference on {len(samples)} samples...")
        results = []
        
        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i : i + batch_size]
            results.extend(self.inference_batch(batch))

        # --- Post-Processing Logic based on Mode ---
        prev_prog = 0.0
        
        for idx, item in enumerate(results):
            raw = item.get("pred", "")
            try:
                val_str = raw.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
                raw_score = max(-100.0, min(100.0, float(val_str))) / 100.0
            except Exception:
                print(f"[ERR] Failed to parse score for item {idx}: {raw}")
                raw_score = 0.0
            
            curr_progress = 0.0
            hop = 0.0

            if eval_mode == "incremental":
                # Original Logic: raw_score is incremental change
                if idx == 0:
                    curr_progress = raw_score
                else:
                    if raw_score >= 0:
                        curr_progress = prev_prog + (1 - prev_prog) * raw_score
                    else:
                        curr_progress = prev_prog + prev_prog * raw_score
                hop = raw_score # In incremental, Model Output IS the Hop signal

            elif eval_mode == "forward":
                # Forward: Model Output IS Progress (Start -> Current)
                curr_progress = raw_score
                # Hop is the change in progress from previous step
                hop = curr_progress - prev_prog

            elif eval_mode == "backward":
                # Backward: Compare Goal -> Current
                # raw_score is likely negative (Current is worse than Goal)
                # Formula: Progress = 1 + ModelOutput
                curr_progress = 1.0 + raw_score
                # Hop is the change in progress
                hop = curr_progress - prev_prog
            
            # Clamp progress to reasonable bounds [0, 1] for safety? 
            # Not strictly enforcing clamping to allow debugging, but typically progress is 0-1.
            # curr_progress = max(0.0, min(1.0, curr_progress))

            item["hop"] = hop
            item["progress"] = curr_progress
            
            prev_prog = curr_progress

        result_path = run_root / "pred_vllm.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Done. Results saved to {result_path}")

        if visualize:
            print("Generating visualization video...")
            plot_video_reward(run_root)

        return str(run_root)


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    
    MODEL_PATH = "../checkpoints/Robo-Dopamine-GRM-3B"
    TASK_INSTRUCTION = "organize the table"
    BASE_DEMO_PATH = "./demo_table"
    GOAL_IMAGE_PATH = None # "./demo_table/goal_image.png" 
    OUTPUT_ROOT = "../results"
    
    agent = GRMInference(
        model_path=MODEL_PATH,
        max_image_num=8
    )

    print("## =====================================")
    print("## (1) Example for Incremental-Mode")
    print("## =====================================")
    mode = "incremental" # "incremental", "forward", "backward"

    output_dir = agent.run_pipeline(
        cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
        cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
        out_root       = OUTPUT_ROOT,
        task           = TASK_INSTRUCTION,
        frame_interval = 30,
        batch_size     = 10,
        goal_image     = GOAL_IMAGE_PATH,
        eval_mode      = mode,
        visualize      = True
    )
    
    print(f"Episode ({BASE_DEMO_PATH}) processed ({mode}). Output at: {output_dir}")

    print("## =====================================")
    print("## (2) Example for Forward-Mode")
    print("## =====================================")
    mode = "forward" # "incremental", "forward", "backward"

    output_dir = agent.run_pipeline(
        cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
        cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
        out_root       = OUTPUT_ROOT,
        task           = TASK_INSTRUCTION,
        frame_interval = 30,
        batch_size     = 10,
        goal_image     = GOAL_IMAGE_PATH,
        eval_mode      = mode,
        visualize      = True
    )
    
    print(f"Episode ({BASE_DEMO_PATH}) processed ({mode}). Output at: {output_dir}")

    print("## =====================================")
    print("## (3) Example for Backward-Mode")
    print("## =====================================")
    mode = "backward" # "incremental", "forward", "backward"

    output_dir = agent.run_pipeline(
        cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
        cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
        cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
        out_root       = OUTPUT_ROOT,
        task           = TASK_INSTRUCTION,
        frame_interval = 30,
        batch_size     = 10,
        goal_image     = GOAL_IMAGE_PATH,
        eval_mode      = mode,
        visualize      = True
    )
    
    print(f"Episode ({BASE_DEMO_PATH}) processed ({mode}). Output at: {output_dir}")
