import os
import json
import base64
import time
import shutil
import re
import cv2
import numpy as np
import argparse

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from tqdm import tqdm
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# OpenAI API Setup
# -----------------------------
from openai import OpenAI

# OpenAI SDK errors: different versions expose errors differently
try:
    from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
except Exception:
    # fallback for older/newer package layouts
    try:
        from openai.error import Timeout as APITimeoutError
        from openai.error import APIConnectionError, RateLimitError, InternalServerError
    except Exception:
        APITimeoutError = APIConnectionError = RateLimitError = InternalServerError = Exception


# -----------------------------
# Utilities
# -----------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_pngs_sorted(dir_path: Path) -> List[Path]:
    files = sorted([p for p in dir_path.iterdir()
                    if p.is_file() and (p.suffix.lower() == ".png" or p.suffix.lower() == ".jpg")])
    return files

def get_frame_count_from_video(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count from video: {video_path}")
    return n

def detect_source_and_count(path: Path) -> Tuple[str, int]:
    if path.is_dir():
        files = list_pngs_sorted(path)
        if len(files) == 0:
            raise RuntimeError(f"No PNG/JPG frames found in directory: {path}")
        return "dir", len(files)
    else:
        return "video", get_frame_count_from_video(path)

def make_sample_indices(num_frames: int, m: int) -> List[int]:
    if num_frames < 1:
        return []
    if m <= 1:
        return [0]
    return [round(i * (num_frames - 1) / (m - 1)) for i in range(m)]

# -----------------------------
# Frame extraction / copying
# -----------------------------

def save_frames_from_video(video_path: Path, out_dir: Path, indices: List[int]) -> List[int]:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")

        out_path = out_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        saved.append(idx)

    cap.release()
    return saved

def save_frames_from_dir(src_dir: Path, out_dir: Path, indices: List[int]) -> List[int]:
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
    inverse: bool = False,
) -> list:

    ts = run_root.name
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
                    str(run_root / ".cache" / "cam_high"        / f"frame_{0:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{last_idx:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{af:06d}.png"),
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{af:06d}.png"),
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{af:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{bf:06d}.png"),
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{bf:06d}.png"),
                ]
            })
        else:
            items.append({
                "id": f"step-{sample_id}-bf_{bf:06d}-af_{af:06d}",
                "task": task,
                "image": [
                    str(run_root / ".cache" / "cam_high"        / f"frame_{0:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{last_idx:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{bf:06d}.png"),
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{bf:06d}.png"),
                    str(run_root / ".cache" / "cam_high"        / f"frame_{af:06d}.png"),
                    str(run_root / ".cache" / "cam_left_wrist"  / f"frame_{af:06d}.png"),
                    str(run_root / ".cache" / "cam_right_wrist" / f"frame_{af:06d}.png"),
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

    t_high, n_high = detect_source_and_count(p_high)
    t_left, n_left = detect_source_and_count(p_left)
    t_right, n_right = detect_source_and_count(p_right)

    assert n_high == n_left == n_right, (
        f"Frame counts are not equal: cam_high={n_high}, cam_left_wrist={n_left}, cam_right_wrist={n_right}"
    )

    indices = make_sample_indices(n_high, m)

    if len(indices) == 0:
        raise RuntimeError(f"Empty indices (n_high={n_high}, m={m})")
    indices[0] = 0
    indices[-1] = n_high - 1

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

    items = build_samples_json(run_root, task, indices, last_idx=n_high - 1, inverse=inverse)
    with open(run_root / "sample.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Done. Frames: {cache_root}\nJSON: {run_root / 'sample.json'}")
    return run_root


class OpenAIInference:
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None, max_workers: int = 10,
                 max_retries: int = 5, retry_delay: float = 2.0):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.prompt_text_start = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): """

        self.prompt_text_parts = [
            "\n- REFERENCE END — Robot Front Image (task fully completed): ",
            "\n</Task>\n\nBEFORE Robot Front Image: ",
            "\nBEFORE Robot Left Wrist Image: ",
            "\nBEFORE Robot Right Wrist Image: ",
            "\n\nAFTER Robot Front Image: ",
            "\nAFTER Robot Left Wrist Image: ",
            "\nAFTER Robot Right Wrist Image: ",
            """\n\nGoal
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
        ]

    def _extract_score(self, pred_text: str) -> float:
        # return pred in [-1,1]
        try:
            match = re.search(r"<score>(.*?)</score>", pred_text)
            if match:
                score_str = match.group(1).replace("%", "").strip()
                val = float(score_str)
            else:
                matches = re.findall(r"([+-]?\d+)\s*%", pred_text)
                val = float(matches[-1]) if matches else 0.0
            val = min(100.0, max(-100.0, val))
            return val / 100.0
        except Exception:
            return 0.0

    def inference_single(self, sample: dict, task: Optional[str], image_dir: str) -> dict:
        images = sample["image"]
        current_task = task if task is not None else sample.get("task", "")

        encoded_images = []
        for img_path in images:
            full_path = os.path.join(image_dir, img_path) if image_dir else img_path
            encoded_images.append(encode_image(full_path))

        content = []
        content.append({"type": "text", "text": self.prompt_text_start.format(task=current_task)})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[0]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[0]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[1]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[1]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[2]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[2]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[3]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[3]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[4]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[4]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[5]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[5]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[6]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[6]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_images[7]}"}})
        content.append({"type": "text", "text": self.prompt_text_parts[7].format(task=current_task)})

        delay = self.retry_delay
        pred_text = "<score>0%</score>"
        last_err = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                pred_text = response.choices[0].message.content
                last_err = None
                break
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                print(f"[WARN] API error attempt {attempt+1}/{self.max_retries}: {e}. retry in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                last_err = e
                break

        if last_err is not None:
            print(f"[WARN] inference failed for {sample.get('id')} err={last_err}. use fallback {pred_text}")

        new_item = {
            "id": sample["id"],
            "image": sample.get("image"),
            "pred": pred_text,
        }
        return new_item

    def inference_grm_batch(self, task: Optional[str], sample_list: list, image_dir: str) -> list:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.inference_single, s, task, image_dir) for s in sample_list]
            for fut in futures:
                results.append(fut.result())
        return results

    def run_infer_with_video(self, episode_root: str, batch_size: int = 10, inverse: bool = False):
        sample_path = os.path.join(episode_root, "sample.json")
        pred_path = os.path.join(episode_root, "pred_openai.json")
        os.makedirs(episode_root, exist_ok=True)

        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Start. Read {len(data)} items from: {sample_path}")

        data_with_pred = []

        for i in range(0, len(data), batch_size):
            sample_batch = data[i:i + batch_size]
            batch_results = self.inference_grm_batch(task=None, sample_list=sample_batch, image_dir="")
            data_with_pred.extend(batch_results)

            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(data_with_pred, f, ensure_ascii=False, indent=2)

        # Post-processing progress
        if inverse:
            pre_prog = 1.0
            for idx, item in enumerate(data_with_pred[::-1]):
                try:
                    pred = self._extract_score(item.get("pred", ""))
                    item["hop"] = pred
                    if pred >= 0:
                        item["progress"] = pre_prog + (1 - pre_prog) * pred
                    else:
                        item["progress"] = pre_prog + pre_prog * pred
                    pre_prog = item["progress"]
                except Exception as e:
                    print(f"[ERROR] inv progress idx={idx}, id={item.get('id')}: {e}")
                    item["progress"] = "Error"
        else:
            pre_prog = 0.0
            for idx, item in enumerate(data_with_pred):
                try:
                    pred = self._extract_score(item.get("pred", ""))
                    item["hop"] = pred
                    if pred >= 0:
                        item["progress"] = pre_prog + (1 - pre_prog) * pred
                    else:
                        item["progress"] = pre_prog + pre_prog * pred
                    pre_prog = item["progress"]
                except Exception as e:
                    print(f"[ERROR] progress idx={idx}, id={item.get('id')}: {e}")
                    item["progress"] = "Error"

        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(data_with_pred, f, ensure_ascii=False, indent=2)

        print(f"Done. Wrote {len(data_with_pred)} items to: {pred_path}")

    def run_video(
        self,
        total_chunk_num: int,
        video_path_cam_high: str,
        video_path_cam_left_wrist: str,
        video_path_cam_right_wrist: str,
        out_root: str,
        task: str,
        batch_size: int,
        inverse: bool = False,
    ):
        episode_root = process_videos(
            video_path_cam_high,
            video_path_cam_left_wrist,
            video_path_cam_right_wrist,
            total_chunk_num,
            out_root,
            task,
            inverse=inverse,
        )

        self.run_infer_with_video(str(episode_root), batch_size=batch_size, inverse=inverse)

        pred_path = os.path.join(str(episode_root), "pred_openai.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Result json not found at: {pred_path}")

        with open(pred_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        value = [it.get("progress", 0.0) for it in items]
        return value


def count_images_non_recursive(dir_path: str) -> int:
    exts = (".jpg", ".png", ".JPG", ".PNG")
    if not os.path.exists(dir_path):
        return 0
    filenames = os.listdir(dir_path)
    return sum(1 for f in filenames if f.endswith(exts))

def compute_voc(values) -> float:
    values = np.asarray(values, dtype=float)
    T = len(values)
    if T < 2:
        return 0.0
    time_indices = np.arange(T)
    corr, _ = spearmanr(time_indices, values)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def run_one_eval(agent: OpenAIInference, eva_json: str, args, inverse: bool) -> float:
    input_json = os.path.join(args.input_json_dir, f"{eva_json}.json")
    out_root = os.path.join(
        args.out_root_dir,
        eva_json,
        "voc_positive" if not inverse else "voc_negative",
    )
    in_path = Path(input_json).expanduser().resolve()
    print(f"=============== {input_json} | INVERSE={inverse} ===============")

    if not in_path.exists():
        print(f"[WARN] skipping, file not found: {in_path}")
        return 0.0

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sample_path_list = data["sample_path_list"]
    sample_path_task = data["sample_path_task"]

    sum_voc = 0.0
    num_voc = 0
    voc_dict_list = []

    for name, task in tqdm(list(zip(sample_path_list, sample_path_task)), desc=f"{eva_json} inv={inverse}"):
        try:
            high_cam_path = f"{args.base_dir}/{name}/cam_high"
            if not os.path.exists(high_cam_path):
                print(f"[WARN] Path not found: {high_cam_path}")
                continue

            num_imgs = count_images_non_recursive(high_cam_path)
            total_chunk_num = max(2, num_imgs // args.interval)

            left_path = f"{args.base_dir}/{name}/cam_left_wrist"
            right_path = f"{args.base_dir}/{name}/cam_right_wrist"
            if "egodex" in name:
                left_path = high_cam_path
                right_path = high_cam_path

            value = agent.run_video(
                total_chunk_num=total_chunk_num,
                video_path_cam_high=high_cam_path,
                video_path_cam_left_wrist=left_path,
                video_path_cam_right_wrist=right_path,
                out_root=out_root,
                task=task,
                batch_size=args.batch_size,
                inverse=inverse,
            )

            voc = compute_voc(value)
            if not (-1 <= voc <= 1):
                voc = 0.0

            print(f"{name}  VOC: {voc:.4f}")
            voc_dict_list.append({"name": name, "voc": voc})
            sum_voc += voc
            num_voc += 1

        except Exception as e:
            print(f"[ERROR] {name} skipped. err: {e}")

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
                    "model_name": args.model_name,
                    "interval": args.interval,
                    "inverse": inverse,
                    "input_json": str(in_path),
                    "base_dir": args.base_dir,
                    "out_root": out_root,
                    "sum_voc": sum_voc,
                    "num_voc": num_voc,
                    "avg_voc": avg_voc,
                    "base_url": args.base_url,
                    "max_workers": args.max_workers,
                    "batch_size": args.batch_size,
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
    parser = argparse.ArgumentParser(description="Run VOC eval via OpenAI API with inverse True/False and summary.")

    # API
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None, help="If not set, read from env OPENAI_API_KEY")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=2.0)

    # paths
    parser.add_argument("--input_json_dir", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--out_root_dir", type=str, required=True)

    # runtime
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max_workers", type=int, default=16, help="Max parallel workers for OpenAI API calls.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Pool size for OpenAI API calls, set 1000 as default.")

    # eval list
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
    )

    # inverse modes
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

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API key not provided. Use --api_key or set env OPENAI_API_KEY.")

    inverse_modes = [True if m.lower() == "true" else False for m in args.inverse_modes]

    agent = OpenAIInference(
        model_name=args.model_name,
        api_key=api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    summary = {}

    for eva_json in args.evaluation_list:
        summary[eva_json] = {}

        if False in inverse_modes:
            voc_plus = run_one_eval(agent, eva_json, args, inverse=False)
            summary[eva_json]["voc+"] = voc_plus

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
                    "model_name": args.model_name,
                    "base_url": args.base_url,
                    "interval": args.interval,
                    "input_json_dir": args.input_json_dir,
                    "base_dir": args.base_dir,
                    "out_root_dir": args.out_root_dir,
                    "inverse_modes": [("true" if x else "false") for x in inverse_modes],
                    "evaluation_list": args.evaluation_list,
                    "max_workers": args.max_workers,
                    "batch_size": args.batch_size,
                    "max_retries": args.max_retries,
                    "retry_delay": args.retry_delay,
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
