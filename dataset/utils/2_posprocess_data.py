import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

def load_all_episode_items(root_dir: Path) -> list:
    """
    Load all train.json files from all episode subdirectories under root_dir.
    Collect all items into a single list.
    """
    all_items = []
    missing = 0
    processed = 0
    
    # Find all episode subdirectories and their train.json files
    for train_path in root_dir.glob("episode_*/train.json"):
        processed += 1
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_items.extend(data)
            else:
                print(f"[WARN] {train_path} is not a list; wrapping as single item")
                all_items.append(data)
        except Exception as e:
            print(f"[WARN] failed to load {train_path}: {e}")
            missing += 1
    
    if missing:
        print(f"[INFO] Failed to load train.json in {missing} episode(s).")
    print(f"[INFO] Processed {processed} episode directories")
    print(f"[INFO] Loaded total items: {len(all_items)}")
    return all_items


def save_json(items: list, path: Path):
    """
    Save list of items to JSON file with UTF-8 encoding and pretty formatting.
    Create parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}  ({len(items)} items)")


def get_image_key(it: dict) -> Optional[str]:
    """
    Return the key name for image list: prioritize 'images' (list), then 'image' (list).
    Return None if neither exists or is not a list.
    """
    if "images" in it and isinstance(it["images"], list):
        return "images"
    if "image" in it and isinstance(it["image"], list):
        return "image"
    return None


def replace_front_two_images_across_items(
    items: List[dict], prob: float, rng: random.Random
) -> Tuple[int, int, int, int]:
    """
    Replace the first two images of each sample with the first two images from other samples 
    with a given probability.
    
    Conditions:
      - Both target and donor samples must have at least 2 images
      - Donor index != target index
    
    Returns:
      (replaced, not_replaced, eligible, ineligible)
        - replaced: Number of samples actually replaced
        - not_replaced: Eligible samples not replaced (probability not met)
        - eligible: Total eligible samples (>=2 images)
        - ineligible: Ineligible samples (<2 images)
    """
    # Collect indices of donors (samples with >=2 images)
    donors = []
    for idx, it in enumerate(items):
        key = get_image_key(it)
        if key and isinstance(it[key], list) and len(it[key]) >= 2:
            donors.append(idx)

    eligible = 0
    replaced = 0
    not_replaced = 0
    n = len(items)

    for i in range(n):
        key = get_image_key(items[i])
        if not key or not isinstance(items[i][key], list) or len(items[i][key]) < 2:
            continue  # Skip ineligible samples
        eligible += 1

        # Replace with given probability (only if there are enough donors)
        if rng.random() < prob and len(donors) >= 2:
            # Find a donor index different from current index (max 5 attempts)
            for _ in range(5):
                j = donors[rng.randrange(0, len(donors))]
                if j != i:
                    donor_key = get_image_key(items[j])
                    donor_imgs = items[j][donor_key]
                    if isinstance(donor_imgs, list) and len(donor_imgs) >= 2:
                        items[i][key][0] = donor_imgs[0]
                        items[i][key][1] = donor_imgs[1]
                        replaced += 1
                    break
        else:
            not_replaced += 1

    ineligible = n - eligible
    return replaced, not_replaced, eligible, ineligible


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process and augment AgileX training data by merging episodes and replacing images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root-dir", 
        type=Path, 
        default=Path("train_data"),
        help="Root directory containing episode subdirectories (episode_XXX)"
    )
    parser.add_argument(
        "--merged-json", 
        type=Path, 
        default=Path("train_data/train_jsons/finetune_data_wo_replace.json"),
        help="Path to save merged (unprocessed) JSON file"
    )
    parser.add_argument(
        "--final-json", 
        type=Path, 
        default=Path("train_data/train_jsons/finetune_data_final.json"),
        help="Path to save final processed JSON file"
    )
    parser.add_argument(
        "--replace-prob", 
        type=float, 
        default=0.75,
        help="Probability to replace first two images of eligible samples (0.0-1.0)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility (None for system random)"
    )
    
    args = parser.parse_args()
    
    # Validate input parameters
    if not args.root_dir.exists():
        print(f"[ERROR] Root directory {args.root_dir} does not exist!")
        return
    if not (0.0 <= args.replace_prob <= 1.0):
        print(f"[ERROR] Replace probability must be between 0.0 and 1.0 (got {args.replace_prob})")
        return

    # 1) Load and shuffle all items
    items = load_all_episode_items(args.root_dir)
    rng = random.Random(args.seed)
    rng.shuffle(items)  # Reproducible shuffle if seed is set

    # 2) Save merged data (original, unprocessed)
    save_json(items, args.merged_json)

    # 3) Replace first two images with specified probability
    replaced, not_replaced, eligible, ineligible = replace_front_two_images_across_items(
        items, args.replace_prob, rng
    )
    
    # Print replacement statistics
    total = len(items)
    print("\n[STATS] Front-two image replacement summary:")
    print(f"  Total items:           {total}")
    print(f"  Eligible (>=2 images): {eligible}")
    print(f"  Replaced:              {replaced}")
    print(f"  Not replaced:          {not_replaced}")
    print(f"  Ineligible (<2 imgs):  {ineligible}")

    # 4) Save final processed data
    save_json(items, args.final_json)

if __name__ == "__main__":
    main()
