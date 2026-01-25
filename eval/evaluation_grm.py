import os
import json
import cv2
import shutil
import numpy as np
import argparse

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from datetime import datetime
from typing import List, Tuple
from scipy.stats import spearmanr

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_pngs_sorted(dir_path: Path) -> List[Path]:
    """Return lexicographically sorted .png files (case-insensitive) under dir_path."""
    files = sorted([p for p in dir_path.iterdir() if p.is_file() and (p.suffix.lower() == ".png" or p.suffix.lower() == ".jpg")])
    return files

def get_frame_count_from_video(video_path: Path) -> int:
    """Return total frame count of a video file via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count from video: {video_path}")
    return n

def detect_source_and_count(path: Path) -> Tuple[str, int]:
    """
    Detect whether path is a directory of PNG frames ('dir') or a video file ('video'),
    and return (source_type, frame_count).
    """
    if path.is_dir():
        files = list_pngs_sorted(path)
        if len(files) == 0:
            raise RuntimeError(f"No PNG frames found in directory: {path}")
        return "dir", len(files)
    else:
        return "video", get_frame_count_from_video(path)

def make_sample_indices(num_frames: int, m: int) -> List[int]:
    """
    Evenly sample m indices within [0, num_frames-1], inclusive endpoints.
    Always includes first (0) and last (num_frames-1).
    """
    if num_frames < 1:
        return []
    if m <= 1:
        return [0]
    return [round(i * (num_frames - 1) / (m - 1)) for i in range(m)]

# -----------------------------
# Frame extraction / copying
# -----------------------------

def save_frames_from_video(video_path: Path, out_dir: Path, indices: List[int]) -> List[int]:
    """
    Decode specific frame indices from a video and save as PNG:
    frame_{:06d}.png under out_dir.
    Returns the list of successfully saved indices (should match input).
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Some codecs require a second read after seeking
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")

        out_path = out_dir / f"frame_{idx:06d}.png"
        # PNG compression level: 0 (fastest, largest) .. 9 (slowest, smallest)
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        saved.append(idx)

    cap.release()
    return saved

def save_frames_from_dir(src_dir: Path, out_dir: Path, indices: List[int]) -> List[int]:
    """
    Copy specific frames (by positional indices) from a directory of PNGs to out_dir,
    renaming them to frame_{:06d}.png to normalize naming.
    Returns the list of successfully saved indices.
    """
    ensure_dir(out_dir)
    files = list_pngs_sorted(src_dir)
    n = len(files)
    saved = []
    for idx in indices:
        if not (0 <= idx < n):
            raise RuntimeError(f"Index {idx} is out of range for directory {src_dir} (n={n})")
        src = files[idx]
        dst = out_dir / f"frame_{idx:06d}.png"
        shutil.copyfile(src, dst)
        saved.append(idx)
    return saved

# -----------------------------
# JSON assembly
# -----------------------------

def build_samples_json(
    run_root: Path,
    task: str,
    indices: List[int],
    last_idx: int,
    inverse: bool = False
) -> list:
    """
    Build items for sample.json using adjacent pairs based on the shared 'indices'.
    Each item lists:
      - cam_high first and last frames
      - before/after frames for (high, left_wrist, right_wrist)
    """
    ts = run_root.name  # directory name is already the timestamp %y-%m-%d-%H-%M-%S
    items = []
    if len(indices) < 2:
        return items

    for k in range(len(indices) - 1):
        bf = indices[k]
        af = indices[k + 1]
        sample_id = f"{ts}-{k:04d}"
        if inverse:
            items.append({
                "id": f"step-{sample_id}-bf_{bf:06d}-af_{af:06d}_inv",
                "task": task,
                "image": [
                    str(run_root / ".cache" / "cam_high"        / f"frame_{0:06d}.png"),        # first frame (high)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{last_idx:06d}.png"), # last frame (high)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{af:06d}.png"),       # after  (high)
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{af:06d}.png"),       # after  (left)
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{af:06d}.png"),       # after  (right)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{bf:06d}.png"),       # before (high)
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{bf:06d}.png"),       # before (left)
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{bf:06d}.png"),       # before (right)
                ]
            })
        else:
            items.append({
                "id": f"step-{sample_id}-bf_{bf:06d}-af_{af:06d}",
                "task": task,
                "image": [
                    str(run_root / ".cache" / "cam_high"        / f"frame_{0:06d}.png"),        # first frame (high)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{last_idx:06d}.png"), # last frame (high)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{bf:06d}.png"),       # before (high)
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{bf:06d}.png"),       # before (left)
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{bf:06d}.png"),       # before (right)
                    str(run_root / ".cache" / "cam_high"        / f"frame_{af:06d}.png"),       # after  (high)
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{af:06d}.png"),       # after  (left)
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{af:06d}.png"),       # after  (right)
                ]
            })
    return items

# -----------------------------
# Main pipeline
# -----------------------------

def process_videos(
    video_path_cam_high: str,
    video_path_cam_left_wrist: str,
    video_path_cam_right_wrist: str,
    m: int,
    out_root: str,
    task: str,
    inverse: bool = False,
):
    """
    - If a path is a directory: treat it as a directory of PNG frames, sample by sorted order, and copy.
    - If a path is a file: treat it as a video, decode requested frames and save.
    - Save outputs under: out_root/%y-%m-%d-%H-%M-%S/.cache/{cam_*}/frame_{:06d}.png
    - Generate: out_root/%y-%m-%d-%H-%M-%S/sample.json
    - Assert that all three sources share the same total frame count.
    """
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

    p_high = Path(video_path_cam_high)
    p_left = Path(video_path_cam_left_wrist)
    p_right = Path(video_path_cam_right_wrist)

    # Detect source types and total counts
    t_high, n_high = detect_source_and_count(p_high)
    t_left, n_left = detect_source_and_count(p_left)
    t_right, n_right = detect_source_and_count(p_right)

    # Strictly assert equal total frame counts
    assert n_high == n_left == n_right, (
        f"Frame counts are not equal: cam_high={n_high}, cam_left_wrist={n_left}, cam_right_wrist={n_right}"
    )

    # Shared indices across all three streams (include first and last)
    indices = make_sample_indices(n_high, m)
    if 0 not in indices:
        indices[0] = 0
    if (n_high - 1) not in indices:
        indices[-1] = n_high - 1

    # Save frames for each source based on its type
    if t_high == "video":
        save_frames_from_video(p_high, cam_dirs["cam_high"], indices)
    else:
        save_frames_from_dir(p_high, cam_dirs["cam_high"], indices)

    if t_left == "video":
        save_frames_from_video(p_left, cam_dirs["cam_left_wrist"], indices)
    else:
        save_frames_from_dir(p_left, cam_dirs["cam_left_wrist"], indices)

    if t_right == "video":
        save_frames_from_video(p_right, cam_dirs["cam_right_wrist"], indices)
    else:
        save_frames_from_dir(p_right, cam_dirs["cam_right_wrist"], indices)

    # Build sample.json with adjacent pairs based on shared indices
    items = build_samples_json(run_root, task, indices, last_idx=n_high - 1, inverse=inverse)
    with open(run_root / "sample.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Done. Frames saved under: {cache_root}\nJSON: {run_root / 'sample.json'}")

    return run_root


class VLLMWrappedInference():
    def __init__(self, model_name_or_path, max_image_num=8, min_pixels=12544, max_pixels=76800):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        try:
            self.processor.pad_token_id = self.processor.tokenizer.pad_token_id
            self.processor.eos_token_id = self.processor.tokenizer.eos_token_id
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels
        except Exception:
            pass

        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
        )

        self.model_name_or_path = model_name_or_path

    def inference_grm_batch(self, task, sample_list, image_dir):
        prompts_text_and_vision = []
        for sample in sample_list:
            images = []
            # images
            assert ("task" in sample and sample["task"] is not None) or task is not None, "`task` should not be None."
            assert isinstance(sample["image"], list) and len(sample["image"]) == 8, "`images` should be list with length of 8."
            for image in sample["image"]:
                images.append(Image.open(os.path.join(image_dir, image)) if isinstance(image, str) else image)
            task = task if task is not None else sample["task"]
            # texts
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"\nYou are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.\n\n<Task>\n`{task}`\n\nREFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):\n- REFERENCE START — Robot Front Image (task just starting): "},
                        {"type": "image"},
                        {"type": "text", "text": "\n- REFERENCE END — Robot Front Image (task fully completed): "},
                        {"type": "image"},
                        {"type": "text", "text": "\n</Task>\n\nBEFORE Robot Front Image: "},
                        {"type": "image"},
                        {"type": "text", "text": "\nBEFORE Robot Left Wrist Image: "},
                        {"type": "image"},
                        {"type": "text", "text": "\nBEFORE Robot Right Wrist Image: "},
                        {"type": "image"},
                        {"type": "text", "text": "\n\nAFTER Robot Front Image: "},
                        {"type": "image"},
                        {"type": "text", "text": "\nAFTER Robot Left Wrist Image: "},
                        {"type": "image"},
                        {"type": "text", "text": "\nAFTER Robot Right Wrist Image: "},
                        {"type": "image"},
                        {"type": "text", "text": f"\n\nGoal\nCompare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.\n\nProgress Estimation (no formulas)\n1) Calibrate using the references:\n   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”\n   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.\n2) Direction:\n   - AFTER better than BEFORE → positive score.\n   - AFTER worse than BEFORE → negative score.\n   - Essentially the same → 0.\n3) Normalize to an integer percentage in [-100%, +100%]:\n   - For improvements, scale the improvement relative to what remained from BEFORE to END.\n   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.\n   - Clip to [-100%, +100%] and round to the nearest integer percent.\n\nEvaluation Criteria (apply across all three views)\n1) Task Alignment: Evidence directly tied to `{task}`.\n2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.\n3) View-Specific Evidence & Consistency:\n   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.\n   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).\n   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.\n   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.\n4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.\n5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.\n\nOutput Format (STRICT)\nReturn ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:\n<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>\n"},
                    ],
                }
            ]

            vllm_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # merge text and images
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt, 
                    "multi_modal_data": {"image": images}
                }
            )

        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text_and_vision), f"Out({len(outputs)}) != In({len(prompts_text_and_vision)})"

        res_sample_list = []
        for output, item in zip(outputs, sample_list):
            if "conversations" in item:
                new_item = {
                    "id": item["id"],
                    "gt": item["conversations"][-1]["value"],
                    "pred": output.outputs[0].text,
                }
            else:
                new_item = {
                    "id": item["id"],
                    "image": item["image"],
                    "pred": output.outputs[0].text,
                }
            res_sample_list.append(new_item)

        return res_sample_list

    def run_infer_with_video(self, episode_root, batch_size, inverse: bool = False):

        sample_path = os.path.join(episode_root, "sample.json")
        pred_path = os.path.join(episode_root, "pred_vllm.json")
        
        os.makedirs(episode_root, exist_ok=True)

        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Start. Read {len(data)} items from: {sample_path}")

        sample_batch = []
        data_with_pred = []
        pred_times = 0

        for idx, sample in tqdm(enumerate(data), total=len(data)):
            sample_batch.append(sample)

            if idx % batch_size != batch_size - 1 and idx != len(data) - 1:
                continue
            else:
                sample_batch_with_pred = self.inference_grm_batch(
                    task = None, 
                    sample_list = sample_batch,
                    image_dir = ""
                )
                data_with_pred += sample_batch_with_pred
                pred_times += 1
                sample_batch = []

        if inverse:
            pre_prog = 1
            for idx, item in enumerate(data_with_pred[::-1]):
                try:
                    pred = min(100, max(-100, float(item["pred"].split("<score>")[-1].split("</score>")[0].replace('%', "").strip()))) / 100.0
                    item["hop"] = pred

                    if pred >= 0:
                        item["progress"] = pre_prog + (1 - pre_prog) * pred
                    else:
                        item["progress"] = pre_prog + pre_prog * pred

                    pre_prog = item["progress"]
                except Exception as e:
                    print(f"[ERROR] index={idx}, id={item.get('id')}: {e}")
                    item["progress"] = "Error"
        else:
            pre_prog = 0
            for idx, item in enumerate(data_with_pred):
                try:
                    pred = min(100, max(-100, float(item["pred"].split("<score>")[-1].split("</score>")[0].replace('%', "").strip()))) / 100.0
                    item["hop"] = pred

                    if pred >= 0:
                        item["progress"] = pre_prog + (1 - pre_prog) * pred
                    else:
                        item["progress"] = pre_prog + pre_prog * pred

                    pre_prog = item["progress"]
                except Exception as e:
                    print(f"[ERROR] index={idx}, id={item.get('id')}: {e}")
                    item["progress"] = "Error"

        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(data_with_pred, f, ensure_ascii=False, indent=2)

        print(f"Done. Wrote {len(data_with_pred)} items to: {pred_path}")

    def run_video(
            self, 
            total_chunk_num,
            video_path_cam_high,
            video_path_cam_left_wrist,
            video_path_cam_right_wrist,
            out_root,
            task,
            batch_size,
            inverse: bool = False,
        ):

        episode_root = process_videos(
            video_path_cam_high,
            video_path_cam_left_wrist,
            video_path_cam_right_wrist,
            total_chunk_num,
            out_root,
            task,
            inverse,
        )

        self.run_infer_with_video(episode_root, batch_size=batch_size, inverse=inverse)

        pred_path = os.path.join(episode_root, "pred_vllm.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"pred_vllm.json not found at: {pred_path}")

        with open(pred_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        value = []
        for item in items:
            value.append(item["progress"])
        return value

def count_images_non_recursive(dir_path):
    exts = (".jpg", ".png", ".JPG", ".PNG")
    filenames = os.listdir(dir_path)
    cnt = sum(1 for f in filenames if f.endswith(exts))
    return cnt

def compute_voc(values) -> float:
    values = np.asarray(values, dtype=float)
    T = len(values)
    if T < 2:
        raise ValueError("At least 2 frames to be calculate VOC")
    time_indices = np.arange(T)
    corr, _ = spearmanr(time_indices, values)
    return float(corr)

def run_one_eval(agent, eva_json, args, inverse: bool):
    input_json = os.path.join(args.input_json_dir, f"{eva_json}.json")
    out_root = os.path.join(
        args.out_root_dir,
        eva_json,
        "voc_positive" if not inverse else "voc_negative",
    )
    in_path = Path(input_json).expanduser().resolve()
    print(f"=============== {input_json} | INVERSE={inverse} ===============")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sample_path_list = data["sample_path_list"]
    sample_path_task = data["sample_path_task"]

    sum_voc = 0.0
    num_voc = 0
    voc_dict_list = []

    for name, task in tqdm(list(zip(sample_path_list, sample_path_task)), desc=f"{eva_json} inv={inverse}"):
        try:
            num_imgs = count_images_non_recursive(f"{args.base_dir}/{name}/cam_high")
            total_chunk_num = num_imgs // args.interval

            value = agent.run_video(
                total_chunk_num=total_chunk_num,
                video_path_cam_high=f"{args.base_dir}/{name}/cam_high",
                video_path_cam_left_wrist=(
                    f"{args.base_dir}/{name}/cam_left_wrist"
                    if "egodex" not in name
                    else f"{args.base_dir}/{name}/cam_high"
                ),
                video_path_cam_right_wrist=(
                    f"{args.base_dir}/{name}/cam_right_wrist"
                    if "egodex" not in name
                    else f"{args.base_dir}/{name}/cam_high"
                ),
                out_root=out_root,
                task=task,
                batch_size=args.batch_size,
                inverse=inverse,
            )

            try:
                voc = compute_voc(value)
            except Exception:
                voc = 0.0

            if not (-1 <= voc <= 1):
                voc = 0.0

            print(name, "  VOC:", voc)
            voc_dict_list.append({"name": name, "voc": voc})
            sum_voc += voc

        except Exception as e:
            print(f"[ERROR] {name} skipped. err: {e}")

        num_voc += 1

    avg_voc = sum_voc / num_voc if num_voc > 0 else 0.0

    if inverse:
        result_path = Path(out_root) / f"result_interval_{args.interval}_voc-_{avg_voc:.6f}.json"
    else:
        result_path = Path(out_root) / f"result_interval_{args.interval}_voc+_{avg_voc:.6f}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "model_path": args.model_path,
                    "interval": args.interval,
                    "inverse": inverse,
                    "input_json": str(in_path),
                    "base_dir": args.base_dir,
                    "out_root": out_root,
                    "sum_voc": sum_voc,
                    "num_voc": num_voc,
                    "avg_voc": avg_voc,
                },
                "results": voc_dict_list,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {result_path}")
    return avg_voc

def parse_args():
    parser = argparse.ArgumentParser(description="Run VOC evaluation with inverse True/False.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--out_root_dir", type=str, required=True)

    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument(
        "--evaluation_list",
        nargs="+",
        default=[
            "agibot_test_100",
            "droid_oxe_test_100",
            "galaxea_r1lite_test_100",
            "human_egodex_test_100",
            "libero_test_100",
            "robocasa_test_100",
        ],
        help="List of evaluation json names (without .json).",
    )

    parser.add_argument(
        "--inverse_modes",
        nargs="+",
        default=["false", "true"],
        choices=["false", "true"],
        help='Which inverse modes to run. Default runs both: ["false", "true"].',
    )

    return parser.parse_args()

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":

    args = parse_args()

    inverse_modes = []
    for m in args.inverse_modes:
        inverse_modes.append(True if m.lower() == "true" else False)

    agent = VLLMWrappedInference(
        model_name_or_path=args.model_path,
        max_image_num=8,
        min_pixels=12544,
        max_pixels=76800,
    )

    summary = {}

    for eva_json in args.evaluation_list:
        summary[eva_json] = {}

        # INVERSE=False => voc+
        if False in inverse_modes:
            voc_plus = run_one_eval(agent, eva_json, args, inverse=False)
            summary[eva_json]["voc+"] = voc_plus

        # INVERSE=True => voc-
        if True in inverse_modes:
            voc_minus = run_one_eval(agent, eva_json, args, inverse=True)
            summary[eva_json]["voc-"] = voc_minus

    # ---- save summary_voc.json ----
    summary_path = Path(args.out_root_dir) / "summary_voc.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "model_path": args.model_path,
                    "interval": args.interval,
                    "input_json_dir": args.input_json_dir,
                    "base_dir": args.base_dir,
                    "out_root_dir": args.out_root_dir,
                    "inverse_modes": [("true" if x else "false") for x in inverse_modes],
                    "evaluation_list": args.evaluation_list,
                },
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved summary: {summary_path}\n")

    # ---- print final summary ----
    print("================ FINAL SUMMARY (AVG VOC) ================\n")
    for eva_json in args.evaluation_list:
        voc_p = summary.get(eva_json, {}).get("voc+")
        voc_m = summary.get(eva_json, {}).get("voc-")
        print(f"{eva_json:25s}  voc+ = {voc_p if voc_p is not None else 'NA'}   voc- = {voc_m if voc_m is not None else 'NA'}")
    print("\n=========================================================\n")
