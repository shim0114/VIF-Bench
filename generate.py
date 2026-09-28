"""Generate VIF-Bench outputs with the API image models evaluated in the paper.

  model            endpoint                          API key
  nano_banana_pro  gemini-3-pro-image-preview        GEMINI_API_KEY
  nano_banana      gemini-3.1-flash-image-preview    GEMINI_API_KEY
  gpt_image_1_5    gpt-image-1.5                     OPENAI_API_KEY
  gpt_image_1      gpt-image-1                       OPENAI_API_KEY

Each task's images (Image_0, Image_1, ...) and its instruction are sent to the model and the
output is saved as generations/<name>/<task_id>.png, the layout judge.py expects.

--instruction dense|medium|sparse uses the text-converted instructions instead (visual-instruction
images removed), as in the paper's visual- vs text-instruction comparison. The outputs go to
generations/<model>_ti_<level>/ and are judged, like all outputs, against the original task.

Examples:
  python generate.py --data_dir ./data --model nano_banana_pro
  python generate.py --data_dir ./data --model gpt_image_1_5 --subset vi_vs_ti --instruction medium

Re-running skips tasks that already have an output, so an interrupted run can simply be restarted.
"""
import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from common import find_generated, load_tasks

load_dotenv()

MODELS = {
    "nano_banana_pro": ("gemini", "gemini-3-pro-image-preview"),
    "nano_banana": ("gemini", "gemini-3.1-flash-image-preview"),
    "gpt_image_1_5": ("openai", "gpt-image-1.5"),
    "gpt_image_1": ("openai", "gpt-image-1"),
}
_clients = {}


def _client(provider):
    if provider not in _clients:
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise SystemExit("GEMINI_API_KEY is not set. Put it in .env (see .env.example) or export it.")
            from google import genai
            _clients[provider] = genai.Client(api_key=key)
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise SystemExit("OPENAI_API_KEY is not set. Put it in .env (see .env.example) or export it.")
            from openai import OpenAI
            _clients[provider] = OpenAI(api_key=key)
    return _clients[provider]


def generate_gemini(model_id, image_paths, instruction):
    from google import genai
    config = genai.types.GenerateContentConfig(response_modalities=["Text", "Image"],
                                               image_config=genai.types.ImageConfig(aspect_ratio="1:1"))
    r = _client("gemini").models.generate_content(
        model=model_id, contents=[instruction] + [Image.open(p) for p in image_paths], config=config)
    for part in (r.candidates[0].content.parts or []) if r.candidates else []:
        if part.inline_data is not None:
            return Image.open(BytesIO(part.inline_data.data))
    raise RuntimeError("the response contains no image")


def generate_openai(model_id, image_paths, instruction):
    files = [open(p, "rb") for p in image_paths]
    try:
        r = _client("openai").images.edit(model=model_id, image=files, prompt=instruction, size="1024x1024")
    finally:
        for f in files:
            f.close()
    return Image.open(BytesIO(base64.b64decode(r.data[0].b64_json)))


GENERATORS = {"gemini": generate_gemini, "openai": generate_openai}


def load_inputs(data_dir, subset, task_ids, instruction):
    """(task_id, image paths, instruction) for the original tasks or a text-converted variant."""
    tasks = load_tasks(data_dir, subset, task_ids)
    if instruction == "original":
        return [(t["id"], t["images"], t["instruction"]) for t in tasks]
    path = Path(data_dir) / "text_instruction" / f"{instruction}.jsonl"
    ti = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    return [(t["id"], [Path(data_dir) / p for p in ti[t["id"]]["images"]], ti[t["id"]]["instruction"]) for t in tasks]


def generate_task(model, image_paths, instruction, out_path, max_retries):
    provider, model_id = MODELS[model]
    last = None
    for attempt in range(max_retries):
        try:
            img = GENERATORS[provider](model_id, image_paths, instruction)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return "ok"
        except SystemExit:
            raise
        except Exception as e:                          # rate limits, transient errors, empty responses
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt) + random.random())
    return f"failed ({last})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="VIF-Bench dataset root (contains metadata.jsonl and tasks/)")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--instruction", default="original", choices=["original", "dense", "medium", "sparse"],
                    help="original visual-instruction tasks (default) or a text-converted variant")
    ap.add_argument("--output_dir", default="generations")
    ap.add_argument("--name", default=None,
                    help="output folder name (default: <model>, or <model>_ti_<level> for --instruction)")
    ap.add_argument("--subset", default="all", choices=["all", "vi_vs_ti"],
                    help="vi_vs_ti = the 200 tasks used for the visual- vs text-instruction comparison")
    ap.add_argument("--tasks", nargs="*", default=None, help="generate only these task ids, e.g. v6_n2_00")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    name = args.name or (args.model if args.instruction == "original" else f"{args.model}_ti_{args.instruction}")
    out_dir = Path(args.output_dir) / name
    inputs = load_inputs(args.data_dir, args.subset, args.tasks, args.instruction)
    work = [(tid, imgs, instr) for tid, imgs, instr in inputs
            if args.overwrite or find_generated(out_dir, tid) is None]
    print(f"{len(inputs)} tasks | {len(inputs) - len(work)} already generated | {len(work)} to generate "
          f"with {args.model} ({MODELS[args.model][1]}) -> {out_dir}")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate_task, args.model, imgs, instr, out_dir / f"{tid}.png", args.max_retries): tid
                for tid, imgs, instr in work}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"generate[{args.model}]"):
            status = fut.result()
            if status != "ok":
                failures += 1
                tqdm.write(f"{futs[fut]}: {status}")
    if failures:
        print(f"{failures} tasks failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
