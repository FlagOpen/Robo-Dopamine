import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Set, DefaultDict, Dict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

USER_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
{task_description}

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
1) Task Alignment: Evidence directly tied to {task_description}.
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

# -------------------- Tool Functions --------------------

def list_episodes(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.lower().startswith("episode_")])

def load_episode_ids(ep_dir: Path) -> List[str]:
    jf = ep_dir / "sample_frames.json"
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        ids = data
    elif isinstance(data, dict):
        ids = data.get("id_list") or data.get("ids") or data
    else:
        raise ValueError(f"Unsupported JSON in {jf}")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"No ids found in {jf}")
    return ids

def load_task_instructions(ep_dir: Path) -> List[str]:
    """
    Load task instructions from task_instruction.json.
    Returns a list of instruction strings.
    """
    jf = Path(os.path.dirname(ep_dir)) / "task_instruction.json"
    default_instruction = ["perform the task"]
    
    if not jf.exists():

        print(f"Warning: {ep_dir}/task_instruction.json NOT EXIST !!!")
        return default_instruction
    
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle list format: ["instr1", "instr2"]
        if isinstance(data, list):
            return [str(x) for x in data]
        
        # Handle dict format: {"instruction": "...", "instructions": [...]}
        elif isinstance(data, dict):
            if "instructions" in data and isinstance(data["instructions"], list):
                return [str(x) for x in data["instructions"]]
            if "instruction" in data and isinstance(data["instruction"], str):
                return [data["instruction"]]
            
        return default_instruction
    except Exception as e:
        print(f"[WARN] Error loading instructions for {ep_dir.name}: {e}")
        return default_instruction

def build_images_rel(ep_name: str, first_id: str, last_id: str, before_id: str, after_id: str) -> List[str]:
    rel = []
    # Reference Start
    rel.append(f"train_data/{ep_name}/cam_high/frame_{first_id}.jpg")
    # Reference End
    rel.append(f"train_data/{ep_name}/cam_high/frame_{last_id}.jpg")
    # Before images
    for cam in CAMERAS:
        rel.append(f"train_data/{ep_name}/{cam}/frame_{before_id}.jpg")
    # After images
    for cam in CAMERAS:
        rel.append(f"train_data/{ep_name}/{cam}/frame_{after_id}.jpg")
    return rel

def format_score_tag(first_idx: int, before_idx: int, after_idx: int, last_idx: int):
    if before_idx == after_idx:
        perc = 0.0
    elif before_idx < after_idx:
        denom = (last_idx - before_idx)
        perc = 100.0 * (after_idx - before_idx) / denom if denom != 0 else 0.0
    else:
        denom = (before_idx - first_idx)
        perc = 100.0 * (after_idx - before_idx) / denom if denom != 0 else 0.0
    return f"<score>{perc:+.1f}%</score>", perc

def score_bucket_idx(perc: float, score_bins: int) -> int:
    p = max(-100.0, min(100.0, perc))
    w = 200.0 / score_bins
    idx = int((p + 100.0) // w)
    if idx == score_bins:  # p==100
        idx = score_bins - 1
    return idx

def gap_bucket_idx(gap: int, max_gap: int, gap_bins: int) -> int:
    gap = max(1, min(max_gap, gap))
    w = max_gap / gap_bins if gap_bins > 0 else max_gap
    idx = int((gap - 1) // w) if w > 0 else 0
    if idx >= gap_bins:
        idx = gap_bins - 1
    return idx

def generate_candidate_pool(n_ids: int, want: int, rng: random.Random, oversample_factor: int = 80, hard_cap: int = 100000) -> List[Tuple[int, int]]:
    target = min(hard_cap, max(want * oversample_factor, want * 10))
    pairs: Set[Tuple[int, int]] = set()
    tries = 0
    while len(pairs) < target and tries < target * 20:
        i = rng.randrange(0, n_ids)
        j = rng.randrange(0, n_ids)
        if i == j:
            tries += 1
            continue
        if i < j and i != n_ids - 1:
            pairs.add((i, j))
        elif i > j and i != 0:
            pairs.add((i, j))
        tries += 1
    return list(pairs)

def select_by_score_then_gap(pairs: List[Tuple[int, int]], n_ids: int, m: int, score_bins: int, gap_bins: int) -> List[Tuple[int, int]]:
    first_idx, last_idx = 1, n_ids
    max_gap = n_ids - 2
    buckets: DefaultDict[int, DefaultDict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for i, j in pairs:
        before_idx = i + 1
        after_idx = j + 1
        _, perc = format_score_tag(first_idx, before_idx, after_idx, last_idx)
        sidx = score_bucket_idx(perc, score_bins)
        gap = abs(i - j)
        gidx = gap_bucket_idx(gap, max_gap=max_gap if max_gap > 0 else 1, gap_bins=gap_bins)
        buckets[sidx][gidx].append((i, j))

    base_quota = [m // score_bins] * score_bins
    remainder = m - sum(base_quota)
    center = (score_bins - 1) / 2.0
    order = sorted(range(score_bins), key=lambda k: abs(k - center))
    for k in order[:remainder]:
        base_quota[k] += 1

    selected: List[Tuple[int, int]] = []

    for sidx in range(score_bins):
        q_s = base_quota[sidx]
        if q_s <= 0:
            continue
        total_in_bucket = sum(len(lst) for lst in buckets[sidx].values())
        if total_in_bucket == 0:
            continue

        base_gap = [q_s // gap_bins] * gap_bins
        rem_gap = q_s - sum(base_gap)
        gap_sizes = [len(buckets[sidx].get(g, [])) for g in range(gap_bins)]
        gap_order = sorted(range(gap_bins), key=lambda g: gap_sizes[g], reverse=True)
        for g in gap_order[:rem_gap]:
            base_gap[g] += 1

        for gidx in range(gap_bins):
            need = base_gap[gidx]
            cands = buckets[sidx].get(gidx, [])
            if not cands or need <= 0:
                continue
            random.shuffle(cands)
            take = cands[:need]
            selected.extend(take)
            buckets[sidx][gidx] = cands[need:]

    def take_any_from_score_bucket(sidx: int, need: int) -> int:
        if need <= 0:
            return 0
        got = 0
        rotated = list(range(gap_bins))
        random.shuffle(rotated)
        for _ in range(3):
            for gidx in rotated:
                if got >= need:
                    break
                cands = buckets[sidx].get(gidx, [])
                while cands and got < need:
                    selected.append(cands.pop())
                    got += 1
            if got >= need:
                break
        return got

    deficit = m - len(selected)
    if deficit > 0:
        for sidx in order:
            if deficit <= 0:
                break
            got = take_any_from_score_bucket(sidx, deficit)
            deficit -= got

    if deficit > 0:
        leftovers = []
        for sidx in range(score_bins):
            for gidx in range(gap_bins):
                leftovers.extend(buckets[sidx].get(gidx, []))
        random.shuffle(leftovers)
        selected.extend(leftovers[:deficit])

    uniq = []
    seen: Set[Tuple[int, int]] = set()
    for p in selected:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
        if len(uniq) >= m:
            break
    return uniq

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def make_zero_score_pairs(ids: List[str], k_zero: int, rng: random.Random) -> List[Tuple[int, int, str, str]]:
    n = len(ids)
    if n == 0 or k_zero <= 0:
        return []
    first_num = int(ids[0])
    last_num = int(ids[-1])
    chosen = []
    for _ in range(k_zero):
        i_before = rng.randrange(0, n)
        before_id_str = ids[i_before]
        tmp_low_str = ids[max(0, i_before-1)]
        tmp_high_str = ids[min(n-1, i_before+1)]
        before_num = int(before_id_str)
        tmp_low_num = int(tmp_low_str)
        tmp_high_num = int(tmp_high_str)
        delta = max(1, int(math.floor((tmp_high_num - tmp_low_num) * 0.075)))
        sign = 1 if rng.random() < 0.5 else -1
        after_num = clamp(before_num + sign * delta, first_num, last_num)
        after_id_str = str(after_num).zfill(len(before_id_str))
        chosen.append((i_before, i_before, before_id_str, after_id_str))
    return chosen

def compute_bins_report(ids: List[str], chosen_pairs: List[Tuple[int, int]], score_bins: int, gap_bins: int) -> Dict:
    n_ids = len(ids)
    first_idx, last_idx = 1, n_ids
    max_gap = n_ids - 2 if n_ids >= 3 else 1
    w = 200.0 / score_bins
    score_ranges = [(round(-100 + k * w, 2), round(-100 + (k + 1) * w, 2)) for k in range(score_bins)]
    score_hist = [0] * score_bins
    gap_hists = [[0] * gap_bins for _ in range(score_bins)]
    for i, j in chosen_pairs:
        before_idx = i + 1
        after_idx = j + 1
        _, perc = format_score_tag(first_idx, before_idx, after_idx, last_idx)
        sidx = score_bucket_idx(perc, score_bins)
        gap = abs(i - j)
        gidx = gap_bucket_idx(gap, max_gap=max_gap if max_gap > 0 else 1, gap_bins=gap_bins)
        score_hist[sidx] += 1
        gap_hists[sidx][gidx] += 1
    total_nonzero = len(chosen_pairs)
    score_ratio = [(cnt, (cnt / total_nonzero * 100.0) if total_nonzero > 0 else 0.0) for cnt in score_hist]
    gap_ratio_in_score = []
    for sidx in range(score_bins):
        cnt_s = score_hist[sidx]
        row = []
        for gidx in range(gap_bins):
            cnt = gap_hists[sidx][gidx]
            row.append((cnt, (cnt / cnt_s * 100.0) if cnt_s > 0 else 0.0))
        gap_ratio_in_score.append(row)
    return {
        "total_nonzero": total_nonzero,
        "score_ranges": score_ranges,
        "score_hist": score_hist,
        "score_ratio": score_ratio,
        "gap_hists": gap_hists,
        "gap_ratio_in_score": gap_ratio_in_score
    }

def write_info_txt(ep_dir: Path, ep_name: str, ids: List[str], chosen_pairs: List[Tuple[int, int]], zero_count: int, score_bins: int, gap_bins: int):
    info = compute_bins_report(ids, chosen_pairs, score_bins, gap_bins)
    lines = []
    lines.append(f"Episode: {ep_name}")
    lines.append(f"Frames (id_list) count: {len(ids)}")
    lines.append(f"Samples: non-zero={info['total_nonzero']}, zero={zero_count}, total={info['total_nonzero'] + zero_count}")
    lines.append("")
    lines.append(f"[Score bins] (bins={score_bins}, range per bin on [-100,100])")
    for k, ((low, high), (cnt, pct)) in enumerate(zip(info["score_ranges"], info["score_ratio"])):
        rlabel = f"[{low}, {high})" if k < score_bins - 1 else f"[{low}, {high}]"
        lines.append(f"  - bin {k:02d} {rlabel}: {cnt} ({pct:.2f}%)")
    lines.append("")
    lines.append(f"[Gap bins INSIDE each score bin] (gap_bins={gap_bins}, gaps uniformly split on [1, max_gap])")
    for sidx in range(score_bins):
        cnt_s = info["score_hist"][sidx]
        if cnt_s == 0:
            continue
        (low, high) = info["score_ranges"][sidx]
        rlabel = f"[{low}, {high})" if sidx < score_bins - 1 else f"[{low}, {high}]"
        lines.append(f"  · score bin {sidx:02d} {rlabel} -> {cnt_s} samples:")
        for gidx in range(gap_bins):
            cnt, pct = info["gap_ratio_in_score"][sidx][gidx]
            lines.append(f"      gap bin {gidx}: {cnt} ({pct:.2f}%)")
    lines.append("")
    out_path = ep_dir / "info.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -------------------- Sub-process for single episode --------------------

def process_episode(ep_dir: Path, m: int, seed: int, score_bins: int, gap_bins: int, oversample_factor: int, zero_ratio: float) -> Tuple[str, int]:
    """
    Returns (episode_name, item_count). Writes train.json and info.txt inside the subprocess.
    Note: train_id for each episode starts from 0000000.
    """
    ep_name = ep_dir.name
    # Ensure parallel reproducibility: derive a stable random seed (base_seed + hash)
    rng = random.Random((hash(ep_name) ^ seed) & 0xFFFFFFFF)

    try:
        ids = load_episode_ids(ep_dir)
    except Exception as e:
        print(f"[WARN] Skip {ep_name}: {e}")
        return ep_name, 0

    # Load instructions
    instructions = load_task_instructions(ep_dir)

    n_ids = len(ids)
    if n_ids < 3:
        print(f"[WARN] Skip {ep_name}: too few ids ({n_ids})")
        return ep_name, 0

    first_id = ids[0]
    last_id = ids[-1]

    zero_n = int(round(m * max(0.0, min(1.0, zero_ratio))))
    nonzero_m = max(0, m - zero_n)

    # non-zero sample
    chosen_pairs: List[Tuple[int, int]] = []
    if nonzero_m > 0:
        cand_pairs = generate_candidate_pool(n_ids=n_ids, want=nonzero_m, rng=rng, oversample_factor=oversample_factor)
        if not cand_pairs:
            print(f"[WARN] {ep_name}: failed to create candidate pairs; non-zero part skipped")
        else:
            chosen_pairs = select_by_score_then_gap(pairs=cand_pairs, n_ids=n_ids, m=nonzero_m, score_bins=score_bins, gap_bins=gap_bins)

    # zero sample
    zero_pairs_detail = make_zero_score_pairs(ids=ids, k_zero=zero_n, rng=rng)

    ep_items = []
    train_counter = 0

    # Process non-zero samples
    for i_before, i_after in chosen_pairs:
        before_id = ids[i_before]
        after_id = ids[i_after]
        first_idx = 1
        before_idx = i_before + 1
        after_idx = i_after + 1
        last_idx = n_ids
        score_tag, perc_score = format_score_tag(first_idx, before_idx, after_idx, last_idx)
        images = build_images_rel(ep_name, first_id, last_id, before_id, after_id)
        perc_tag = "plus" if perc_score > 0 else ("minus" if perc_score < 0 else "zero") 

        item_id = f"train_data_{ep_name}-train-bf_{before_id}-af_{after_id}-{perc_tag}-{str(train_counter).zfill(7)}"
        
        # Randomly select a task instruction
        selected_instruction = rng.choice(instructions)

        ep_items.append({
            "id": item_id,
            "image": images,
            "task": selected_instruction,
            "conversations": [
                {"value": USER_PROMPT.format(task_description=selected_instruction), "from": "human"},
                {"value": score_tag, "from": "gpt"}
            ]
        })
        train_counter += 1

    # Process zero samples
    for _i_before, _i_after_dummy, before_id_str, after_id_str in zero_pairs_detail:
        images = build_images_rel(ep_name, first_id, last_id, before_id_str, after_id_str)
        perc_tag = "zero_extra"
        item_id = f"train_data_{ep_name}-train-bf_{before_id_str}-af_{after_id_str}-{perc_tag}-{str(train_counter).zfill(7)}"
        
        # Randomly select a task instruction
        selected_instruction = rng.choice(instructions)
        
        ep_items.append({
            "id": item_id,
            "image": images,
            "task": selected_instruction,
            "conversations": [
                {"value": USER_PROMPT.format(task_description=selected_instruction), "from": "human"},
                {"value": "<score>0.0%</score>", "from": "gpt"}
            ]
        })
        train_counter += 1

    # Write train.json
    ep_out_path = ep_dir / "train.json"
    with open(ep_out_path, "w", encoding="utf-8") as f:
        json.dump(ep_items, f, ensure_ascii=False, indent=2)

    # Write info.txt (Only stats for non-0% score/gap distribution)
    write_info_txt(ep_dir=ep_dir, ep_name=ep_name, ids=ids, chosen_pairs=chosen_pairs, zero_count=len(zero_pairs_detail), score_bins=score_bins, gap_bins=gap_bins)

    return ep_name, len(ep_items)

# -------------------- Main Function (Parallel Scheduling + Merge) --------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel per-episode train.json generation with score-balanced sampling and zero-mix.")
    parser.add_argument("--base-dir", default="train_data", help="Episodes root")
    parser.add_argument("--max_sample_num", type=int, required=True, help="Samples per episode (TOTAL, including zeros)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--output", default=None, help="(Optional) Global merged train.json path; if omitted, only per-episode files are written")
    parser.add_argument("--score-bins", type=int, default=25, help="Number of score buckets across [-100,100]")
    parser.add_argument("--gap-bins", type=int, default=4, help="Number of gap buckets inside each score bucket")
    parser.add_argument("--oversample-factor", type=int, default=100, help="Candidate pool oversampling factor per episode")
    parser.add_argument("--zero-ratio", type=float, default=0.05, help="Proportion of zero-score samples per episode, e.g., 0.10")
    parser.add_argument("--workers", type=int, default=64, help="Number of parallel processes")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    episodes = list_episodes(base_dir)
    if not episodes:
        print(f"[ERROR] No episode_* dirs in {base_dir}")
        return

    print(f"Found {len(episodes)} episodes. Spawning {args.workers} worker(s).")

    results = []
    # Parallel processing
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {
            ex.submit(
                process_episode,
                ep_dir,
                args.max_sample_num,
                args.seed,
                args.score_bins,
                args.gap_bins,
                args.oversample_factor,
                args.zero_ratio
            ): ep_dir for ep_dir in episodes
        }
        for fut in as_completed(future_map):
            ep_dir = future_map[fut]
            try:
                ep_name, cnt = fut.result()
                print(f"[DONE] {ep_name}: {cnt} items")
                results.append((ep_name, cnt))
            except Exception as e:
                print(f"[ERROR] {ep_dir.name}: {e}")

    # Optional: Merge logic
    if args.output:
        merged_out_path = Path(args.output).expanduser().resolve()
        merged_out_path.parent.mkdir(parents=True, exist_ok=True)
        merged = []
        for ep_dir in episodes:
            ep_train = ep_dir / "train.json"
            if ep_train.exists():
                try:
                    with open(ep_train, "r", encoding="utf-8") as f:
                        merged.extend(json.load(f))
                except Exception as e:
                    print(f"[WARN] Skip merging {ep_train}: {e}")
        with open(merged_out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {merged_out_path}  (merged {len(merged)} items across {len(episodes)} episodes)")

    # Summary
    total_items = sum(cnt for _, cnt in results)
    print("\nSummary:")
    for ep_name, cnt in sorted(results):
        print(f" - {ep_name}: {cnt} items")
    print(f"TOTAL: {total_items} items from {len(results)} episodes")

if __name__ == "__main__":
    main()