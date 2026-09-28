"""Summarize VIF-Bench scores (the numbers reported in the paper).

  python summarize.py --data_dir ./data --results_dir ./results
  python summarize.py --data_dir ./data --results_dir ./results --by vi_type
  python summarize.py --data_dir ./data --results_dir ./results --by conflict --metric visual_instruction_adherence

The main score of the paper is the ensemble of the GPT-5 and Gemini 2.5 Flash judges: for every
task the two judges' scores are averaged, then averaged over tasks (only tasks scored by both
judges are used). Qwen3-VL scores are reported separately. Each model directory under
--results_dir is one row of the table.
"""
import argparse
import json
from pathlib import Path

from common import METRIC_KEYS, load_tasks

SHORT = {"text_instruction_following": "Text Instr.", "reference_consistency": "Ref. Consist.",
         "visual_instruction_adherence": "VI Adherence", "visual_instruction_cleanliness": "VI Cleanliness",
         "scene_coherence": "Scene Coher.", "visual_quality": "Visual Quality", "average": "Avg."}
VI_TYPES = ["layout", "orientation", "light", "wind", "pose"]


def load_scores(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r} if path.exists() else {}


def judge_scores(model_dir, judges):
    """Per-task scores for one model; several judges are averaged per task (tasks scored by all of them)."""
    per = [load_scores(model_dir / j / "scores.jsonl") for j in judges]
    if not all(per):
        return {}
    ids = set.intersection(*[set(p) for p in per])
    return {i: {k: sum(p[i][k] for p in per) / len(per) for k in METRIC_KEYS + ["average"]} for i in ids}


def has_type(meta, t):
    return any(t == x or (t in ("light", "wind") and x == "light+wind") for x in meta["visual_instruction_types"])


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--judges", default="gpt,gemini",
                    help="comma-separated judges averaged per task: gpt,gemini (paper) | qwen3vl | gpt | gemini")
    ap.add_argument("--by", default="overall", choices=["overall", "vi_type", "conflict"])
    ap.add_argument("--metric", default="average", help="metric for --by vi_type / conflict")
    ap.add_argument("--subset", default="all", choices=["all", "vi_vs_ti"])
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args()

    judges = args.judges.split(",")
    meta = {t["id"]: t["meta"] for t in load_tasks(args.data_dir, args.subset)}
    models = sorted(d for d in Path(args.results_dir).iterdir() if d.is_dir())
    table = []
    for d in models:
        s = {i: v for i, v in judge_scores(d, judges).items() if i in meta}
        if not s:
            continue
        rows = list(s.values())
        if args.by == "overall":
            row = {"model": d.name, "n_tasks": len(rows), **{k: mean(rows, k) for k in METRIC_KEYS + ["average"]}}
        elif args.by == "vi_type":
            row = {"model": d.name}
            for t in VI_TYPES:
                row[t] = mean([v for i, v in s.items() if has_type(meta[i], t)], args.metric)
        else:
            row = {"model": d.name}
            for t in VI_TYPES[1:]:
                sub = [(i, v) for i, v in s.items() if has_type(meta[i], t)]
                row[f"{t}_conflict"] = mean([v for i, v in sub if meta[i]["conflicts"][f"{t}_conflict"]], args.metric)
                row[f"{t}_no_conflict"] = mean([v for i, v in sub if not meta[i]["conflicts"][f"{t}_conflict"]], args.metric)
        table.append(row)
    if not table:
        raise SystemExit(f"no scores found for judges {judges} under {args.results_dir}")

    cols = [c for c in table[0] if c != "model"]
    head = [SHORT.get(c, c) for c in cols]
    title = f"judges = {' + '.join(judges)}" + ("" if args.by == "overall" else f", metric = {args.metric}")
    print(f"\n{title}\n")
    print("| Model | " + " | ".join(head) + " |")
    print("|---|" + "---|" * len(cols))
    for r in table:
        print(f"| {r['model']} | " + " | ".join(str(r[c]) if c == "n_tasks" else f"{r[c]:.2f}" for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("model," + ",".join(cols) + "\n")
            for r in table:
                f.write(r["model"] + "," + ",".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
