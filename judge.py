"""Evaluate generated images on VIF-Bench with GPT-5 or Gemini 2.5 Flash as the judge.

API keys are read from environment variables (or a local .env file, see .env.example):
  OPENAI_API_KEY   for --judge gpt
  GEMINI_API_KEY   for --judge gemini

Example:
  python judge.py --data_dir ./data --generated_dir ./generations/my_model --judge gpt
  python judge.py --data_dir ./data --generated_dir ./generations/my_model --judge gemini

Each response is saved to results/<model_name>/<judge>/responses/<task_id>.txt and the
parsed scores to results/<model_name>/<judge>/scores.jsonl. Re-running skips tasks that already
have a valid response, so an interrupted run can simply be restarted.
"""
import argparse
import base64
import mimetypes
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from common import extract_scores, find_generated, load_tasks, response_path, write_scores
from prompts import PROMPT_CRITERIA, PROMPT_HEADER, build_middle

load_dotenv()

# endpoint versions used in the paper
GPT_JUDGE_MODEL = "gpt-5-2025-08-07"
GEMINI_JUDGE_MODEL = "gemini-2.5-flash"

_clients = {}


def _openai_client():
    if "gpt" not in _clients:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set. Put it in .env (see .env.example) or export it.")
        from openai import OpenAI
        _clients["gpt"] = OpenAI(api_key=key)
    return _clients["gpt"]


def _gemini_client():
    if "gemini" not in _clients:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set. Put it in .env (see .env.example) or export it.")
        from google import genai
        _clients["gemini"] = genai.Client(api_key=key)
    return _clients["gemini"]


def _data_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('utf-8')}"


def judge_with_gpt(image_paths, instruction, generated_path):
    content = [{"type": "text", "text": PROMPT_HEADER}]
    content += [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in image_paths]
    content.append({"type": "text", "text": build_middle(instruction)})
    content.append({"type": "image_url", "image_url": {"url": _data_url(generated_path)}})
    content.append({"type": "text", "text": PROMPT_CRITERIA})
    r = _openai_client().chat.completions.create(
        model=GPT_JUDGE_MODEL, messages=[{"role": "user", "content": content}], max_completion_tokens=8192)
    return r.choices[0].message.content or ""


def judge_with_gemini(image_paths, instruction, generated_path):
    contents = [PROMPT_HEADER] + [Image.open(p) for p in image_paths]
    contents += [build_middle(instruction), Image.open(generated_path), PROMPT_CRITERIA]
    r = _gemini_client().models.generate_content(model=GEMINI_JUDGE_MODEL, contents=contents)
    return "".join(p.text or "" for p in r.candidates[0].content.parts)


JUDGES = {"gpt": judge_with_gpt, "gemini": judge_with_gemini}


def judge_task(task, generated, judge, out_path, max_retries):
    """Query the judge until the response contains all six scores (with backoff on API errors)."""
    last = None
    for attempt in range(max_retries):
        try:
            text = JUDGES[judge](task["images"], task["instruction"], generated)
            if extract_scores(text) is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text, encoding="utf-8")
                return "ok"
            last = "unparsable response"
        except SystemExit:
            raise
        except Exception as e:                          # rate limits, transient server errors
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt) + random.random())
    return f"failed ({last})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="VIF-Bench dataset root (contains metadata.jsonl and tasks/)")
    ap.add_argument("--generated_dir", required=True, help="directory with <task_id>.png generated images")
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--model_name", default=None, help="name used in the output path (default: generated_dir name)")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--subset", default="all", choices=["all", "vi_vs_ti"],
                    help="vi_vs_ti = the 200 tasks used for the visual- vs text-instruction comparison")
    ap.add_argument("--tasks", nargs="*", default=None, help="evaluate only these task ids, e.g. v6_n2_00")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model_name = args.model_name or Path(args.generated_dir).resolve().name
    tasks = load_tasks(args.data_dir, args.subset, args.tasks)
    work, missing = [], 0
    for t in tasks:
        g = find_generated(args.generated_dir, t["id"])
        if g is None:
            missing += 1
            continue
        out = response_path(args.output_dir, model_name, args.judge, t["id"])
        if out.exists() and not args.overwrite and extract_scores(out.read_text(encoding="utf-8")) is not None:
            continue
        work.append((t, g, out))
    print(f"{len(tasks)} tasks | {missing} without a generated image | {len(work)} to judge with {args.judge}")

    if work:
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_task, t, g, args.judge, out, args.max_retries): t["id"] for t, g, out in work}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"judge[{args.judge}]"):
                status = fut.result()
                if status != "ok":
                    failures += 1
                    tqdm.write(f"{futs[fut]}: {status}")
        if failures:
            print(f"{failures} tasks failed; re-run the same command to retry them.")

    write_scores(args.output_dir, model_name, args.judge, tasks)


if __name__ == "__main__":
    main()
