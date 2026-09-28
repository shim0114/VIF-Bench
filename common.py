"""Shared helpers: task loading, locating generated images, score parsing and result files."""
import json
import re
from pathlib import Path

# names as they appear in the judge prompt -> keys used in the result files
METRICS = [
    ("Text Instruction Following", "text_instruction_following"),
    ("Reference Consistency", "reference_consistency"),
    ("Vision Instruction Adherence", "visual_instruction_adherence"),
    ("Visual Instruction Residue", "visual_instruction_cleanliness"),
    ("Scene Coherence", "scene_coherence"),
    ("Visual Quality", "visual_quality"),
]
METRIC_KEYS = [k for _, k in METRICS]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def load_tasks(data_dir, subset="all", task_ids=None):
    """Read metadata.jsonl of the dataset.

    Returns a list of dicts: id ("v6_n2_00"), images (paths of Image_0 ... Image_N in the order
    given to the generator), instruction, and the raw metadata row.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found: --data_dir must point to the VIF-Bench dataset root")
    wanted = set(task_ids) if task_ids else None
    tasks = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if wanted is not None and m["id"] not in wanted:
            continue
        if subset == "vi_vs_ti" and not m.get("in_vi_vs_ti_experiment"):
            continue
        images = [data_dir / im["file"] for im in sorted(m["images"], key=lambda im: im["index"])]
        tasks.append(dict(id=m["id"], images=images, instruction=m["instruction"], meta=m))
    return tasks


def find_generated(generated_dir, task_id):
    """Generated image of a task: <generated_dir>/<task_id>.{png,jpg,jpeg,webp}."""
    base = Path(generated_dir) / task_id
    for ext in IMAGE_EXTS:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def extract_scores(text):
    """Parse the six 'Name: <1-10>' lines of a judge response. None if any is missing."""
    scores = {}
    for name, key in METRICS:
        hits = re.findall(rf"{re.escape(name)}\**\s*:\s*\**\s*(\d+(?:\.\d+)?)", text or "")
        if not hits:
            return None
        v = float(hits[-1])          # the final assessment block comes last
        if not 1 <= v <= 10:
            return None
        scores[key] = v
    return scores


def result_dir(output_dir, model_name, judge_name):
    return Path(output_dir) / model_name / judge_name


def response_path(output_dir, model_name, judge_name, task_id):
    return result_dir(output_dir, model_name, judge_name) / "responses" / f"{task_id}.txt"


def write_scores(output_dir, model_name, judge_name, tasks):
    """Collect every saved response into <output_dir>/<model>/<judge>/scores.jsonl."""
    out = result_dir(output_dir, model_name, judge_name)
    n_ok = n_bad = 0
    rows = []
    for t in tasks:
        p = response_path(output_dir, model_name, judge_name, t["id"])
        if not p.exists():
            continue
        s = extract_scores(p.read_text(encoding="utf-8"))
        if s is None:
            n_bad += 1
            continue
        n_ok += 1
        rows.append(dict(id=t["id"], **s, average=sum(s.values()) / len(s)))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{model_name} / {judge_name}] {n_ok} scored, {n_bad} unparsable -> {out / 'scores.jsonl'}")
    if rows:
        means = {k: sum(r[k] for r in rows) / len(rows) for k in METRIC_KEYS + ["average"]}
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in means.items()))
    return rows
